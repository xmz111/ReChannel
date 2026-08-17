"""Native PyTorch training for ReChannel 4B.

Single GPU:
  python training/train.py --task normal --data-root /datasets/hypersim_normal
Multi GPU:
  torchrun --nproc_per_node=2 training/train.py --task depth --data-root /datasets/depth

Data layouts (no data is downloaded by this script):
  depth:  <root>/hypersim_hf/train/**/rgb_*.png + depth_plane_*.png,
          optionally <root>/vkitti/**/rgb_*.jpg + matching depth/*.png
  normal: <root>/**/*.npz, each with color (H,W,3 in [0,1]) and normal (H,W,3)
  matting:<root>/*_img.jpg and matching *_alpha.png
  refseg: <root>/train2014/COCO_train2014_*.jpg and <root>/refs/*_train.parquet
"""
from __future__ import annotations

import argparse, glob, json, os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from diffusers import Flux2KleinPipeline
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file

WEIGHTS_REPO = "xmz111/ReChannel"
BACKBONE = "black-forest-labs/FLUX.2-klein-base-4B"
LORA_TARGETS = ("to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj",
                "add_v_proj", "to_add_out", "to_qkv_mlp_proj", "linear_in", "linear_out")


@dataclass(frozen=True)
class Task:
    channels: int
    loss: str
    text: bool = False


TASKS = {
    "depth": Task(1, "silog"),
    "normal": Task(3, "cosine"),
    "matting": Task(1, "mat_l1"),
    "refseg": Task(1, "bce", text=True),
}

# One public reference recipe.  Tasks differ only in data, output, loss and text condition.
RANK, ALPHA, BATCH, STEPS, LR = 64, 32, 16, 15_000, 1e-4
EMA_DECAY, EMA_WARMUP, SAVE_EVERY = .999, 2_000, 1_000


class ThinPixelTail(nn.Module):
    """One token-local linear layer; no convolution or spatial mixing."""
    def __init__(self, channels):
        super().__init__()
        self.unpatch_linear = nn.Linear(128, 16 * 16 * channels)
        self.channels = channels

    def forward(self, tokens):
        b, n, _ = tokens.shape
        h = w = int(n ** .5)
        if h * w != n:
            raise ValueError(f"expected a square token grid, got {n} tokens")
        x = self.unpatch_linear(tokens).reshape(b, h, w, 16, 16, self.channels)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(b, self.channels, h * 16, w * 16)


def patchify_bn(z, mean, std):
    b, c, h, w = z.shape
    z = z.float().permute(0, 2, 3, 1)
    z = z.reshape(b, h // 2, 2, w // 2, 2, c).permute(0, 1, 3, 5, 2, 4)
    return (z.reshape(b, -1, c * 4) - mean) / std


def loss_for(name, pred, target):
    if name == "silog":
        valid = (target > .01) & (target < 100.)
        d = pred[:, 0][valid] - target[valid].log()
        return d.square().mean() - .85 * d.mean().square() if len(d) else pred.sum() * 0
    if name == "cosine":
        valid = target.norm(dim=1) > .5
        cos = (F.normalize(pred, dim=1) * F.normalize(target, dim=1)).sum(1).clamp(-1 + 1e-6, 1 - 1e-6)
        n = valid.sum().clamp_min(1)
        return ((1 - cos) * valid).sum() / n + .1 * (cos.acos() / (torch.pi / 2) * valid).sum() / n
    if name == "mat_l1":
        return (pred[:, 0].tanh() - (target * 2 - 1)).abs().mean()
    return F.binary_cross_entropy_with_logits(pred[:, 0], (target + 1) / 2)


def rgb(path, res):
    x = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB).astype("float32")
    return cv2.resize(x, (res, res), interpolation=cv2.INTER_LINEAR) / 127.5 - 1


class DenseDataset(Dataset):
    def __init__(self, task, root, res=512):
        self.task, self.root, self.res = task, Path(root), res
        if task == "depth":
            self.items = []
            for depth in glob.glob(str(self.root / "hypersim_hf/train/**/*depth_plane*.png"), recursive=True):
                image = depth.replace("depth_plane_", "rgb_")
                if os.path.exists(image): self.items.append(("hs", image, depth))
            for image in glob.glob(str(self.root / "vkitti/**/rgb_*.jpg"), recursive=True):
                depth = image.replace("/rgb/", "/depth/").replace("rgb_", "depth_").replace(".jpg", ".png")
                if os.path.exists(depth): self.items.append(("vk", image, depth))
        elif task == "normal":
            self.items = glob.glob(str(self.root / "**/*.npz"), recursive=True)
        elif task == "matting":
            self.items = [(p, p.replace("_img.jpg", "_alpha.png")) for p in glob.glob(str(self.root / "*_img.jpg"))]
            self.items = [(a, b) for a, b in self.items if os.path.exists(b)]
        else:
            import pyarrow.parquet as pq
            self.items = []
            for p in glob.glob(str(self.root / "refs/*_train.parquet")):
                for row in pq.read_table(p).to_pylist():
                    info = json.loads(row["raw_image_info"])
                    image = self.root / "train2014" / f"COCO_train2014_{row['image_id']:012d}.jpg"
                    if image.exists():
                        self.items += [(image, s["sent"] if isinstance(s, dict) else str(s), row["raw_anns"], info["height"], info["width"])
                                       for s in row["sentences"]]
        if not self.items: raise RuntimeError(f"No {task} samples under {self.root}")
        print(f"[{task}] {len(self.items)} samples")

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        if self.task == "depth":
            kind, image, depth = self.items[i]; x = rgb(image, self.res)
            y = cv2.imread(depth, cv2.IMREAD_ANYDEPTH).astype("float32")
            y = y / (1000 if kind == "hs" else 100)
            return x, cv2.resize(y, (self.res, self.res), interpolation=cv2.INTER_NEAREST)
        if self.task == "normal":
            d = np.load(self.items[i]); x, y = d["color"].astype("float32"), d["normal"].astype("float32")
            x = cv2.resize(x, (self.res, self.res)) * 2 - 1
            y = cv2.resize(y, (self.res, self.res), interpolation=cv2.INTER_NEAREST)
            return x, y / (np.linalg.norm(y, axis=-1, keepdims=True) + 1e-8)
        if self.task == "matting":
            image, alpha = self.items[i]
            y = cv2.imread(alpha, cv2.IMREAD_GRAYSCALE).astype("float32") / 255
            return rgb(image, self.res), cv2.resize(y, (self.res, self.res))
        image, phrase, raw, h, w = self.items[i]
        from pycocotools import mask as masks
        ann = json.loads(raw); seg = ann["segmentation"] if isinstance(ann, dict) else ann[0]["segmentation"]
        rle = masks.merge(masks.frPyObjects(seg, h, w)) if isinstance(seg, list) else masks.frPyObjects(seg, h, w) if isinstance(seg["counts"], list) else seg
        y = cv2.resize(masks.decode(rle).astype("float32"), (self.res, self.res), interpolation=cv2.INTER_NEAREST)
        return rgb(image, self.res), y * 2 - 1, phrase


def collate(batch):
    x = torch.from_numpy(np.stack([v[0] for v in batch])).permute(0, 3, 1, 2).float()
    y = torch.from_numpy(np.stack([v[1] for v in batch]))
    if y.ndim == 4: y = y.permute(0, 3, 1, 2)
    return (x, y.float(), [v[2] for v in batch]) if len(batch[0]) == 3 else (x, y.float(), None)


def text_embed(tokenizer, encoder, phrases, device):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    rendered = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False) for p in phrases]
    inputs = tokenizer(rendered, return_tensors="pt", padding="max_length", truncation=True, max_length=32).to(device)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        out = encoder(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    return torch.cat([out.hidden_states[i] for i in (9, 18, 27)], -1).bfloat16()


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--task", choices=TASKS, required=True); ap.add_argument("--data-root", required=True)
    ap.add_argument("--output", default="checkpoints"); ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--batch", type=int, default=BATCH); ap.add_argument("--rank", type=int, default=RANK); ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--save-every", type=int, default=SAVE_EVERY); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-ema", action="store_true"); args = ap.parse_args(); cfg = TASKS[args.task]
    world = int(os.environ.get("WORLD_SIZE", "1")); local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1: dist.init_process_group("nccl"); torch.cuda.set_device(local)
    device = torch.device("cuda", local); primary = local == 0
    rank, batch, steps = args.rank, args.batch, args.steps
    token = os.getenv("HF_TOKEN")
    pipe = Flux2KleinPipeline.from_pretrained(BACKBONE, torch_dtype=torch.bfloat16, token=token)
    vae = pipe.vae.to(device).eval(); [p.requires_grad_(False) for p in vae.parameters()]
    body = get_peft_model(pipe.transformer.to(device, torch.bfloat16), LoraConfig(r=rank, lora_alpha=ALPHA, target_modules=LORA_TARGETS, bias="none"))
    tail = ThinPixelTail(cfg.channels).to(device)
    if cfg.text:
        encoder = pipe.text_encoder.to(device).eval(); [p.requires_grad_(False) for p in encoder.parameters()]
    aux = load_file(hf_hub_download(WEIGHTS_REPO, "auxiliary/aux_4b.safetensors", token=token))
    mean, std, img_ids = (aux[k].to(device) for k in ("bn_mean", "bn_std", "img_ids"))
    null, null_ids = aux["null_emb"].to(device), aux["null_text_ids"].to(device)
    while null.ndim > 3: null = null.squeeze(0)
    while null_ids.ndim > 2: null_ids = null_ids.squeeze(0)
    text_ids = torch.tensor(np.stack([np.zeros(32), np.zeros(32), np.zeros(32), np.arange(32)], -1), device=device, dtype=torch.float32)
    ds = DenseDataset(args.task, args.data_root); sampler = DistributedSampler(ds) if world > 1 else None
    loader = DataLoader(ds, batch_size=batch, shuffle=sampler is None, sampler=sampler, drop_last=True, pin_memory=True, num_workers=args.workers, collate_fn=collate)
    if world > 1: body, tail = DDP(body, device_ids=[local]), DDP(tail, device_ids=[local])
    params = [p for m in (body, tail) for p in m.parameters() if p.requires_grad]; opt = AdamW(params, lr=args.lr)
    named = lambda: [(n, p) for n, p in (body.module if world > 1 else body).named_parameters() if p.requires_grad] + [("tail." + n, p) for n, p in (tail.module if world > 1 else tail).named_parameters() if p.requires_grad]
    ema = None; outdir = Path(args.output) / args.task; outdir.mkdir(parents=True, exist_ok=True); it = iter(loader)
    for step in range(1, steps + 1):
        if sampler and step % len(loader) == 1: sampler.set_epoch(step)
        try: x, y, phrases = next(it)
        except StopIteration: it = iter(loader); x, y, phrases = next(it)
        x, y = x.to(device, torch.bfloat16), y.to(device)
        with torch.no_grad(): z = vae.encode(x).latent_dist.mode()
        cond, ids = (text_embed(pipe.tokenizer, encoder, phrases, device), text_ids) if cfg.text else (null.expand(len(x), -1, -1), null_ids)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            field = body(hidden_states=patchify_bn(z, mean, std).bfloat16(), encoder_hidden_states=cond, timestep=torch.full((len(x),), .999, device=device), img_ids=img_ids, txt_ids=ids, guidance=torch.zeros(len(x), device=device), return_dict=False)[0]
        loss = loss_for(cfg.loss, tail(field.float()), y); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad()
        if not args.no_ema:
            if step == EMA_WARMUP: ema = {n: p.detach().clone() for n, p in named()}
            elif ema: [ema[n].mul_(EMA_DECAY).add_(p.detach(), alpha=1 - EMA_DECAY) for n, p in named()]
        if primary and step % 100 == 0: print(f"step {step}/{steps} loss={loss.item():.4f}", flush=True)
        if primary and step % args.save_every == 0:
            save_file({n: p.detach().cpu().float() for n, p in named()}, outdir / f"step_{step:06d}.safetensors")
            if ema: save_file({n: p.cpu().float() for n, p in ema.items()}, outdir / f"step_{step:06d}_ema.safetensors")
    if world > 1: dist.destroy_process_group()


if __name__ == "__main__": main()
