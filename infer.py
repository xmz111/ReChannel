"""ReChannel — single-image dense-field readout with a frozen text-to-image DiT.

Reads out task-native pixel fields from a frozen FLUX-Klein backbone via a
token-local linear head (no target-side VAE decoder, no spatial mixing).

  input RGB --VAE enc--> tokens --DiT(frozen + task LoRA, sigma=0)--> token field
            --Linear(128 -> p*p*K) + reshape--> p x p x K pixel patch --tile--> output

Pure PyTorch. Weights are pulled from the Hugging Face Hub. This is an *inference /
qualitative demo* script; it is not the benchmark-evaluation pipeline used for the
paper's tables.

Usage:
  python infer.py --image path/to/img.jpg \
      --tasks depth,normal,matting,refseg \
      --phrase "the couch" --out out.png
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from diffusers import Flux2KleinPipeline

WEIGHTS_REPO = "xmz111/ReChannel"                      # per-task LoRA + linear head (bf16, public)
KLEIN_REPO = "black-forest-labs/FLUX.2-klein-base-4B"  # frozen backbone
LORA_SCALE = 0.5                                        # alpha/rank, same for all released tasks

# task -> (weights subfolder, output channels K, patch size, uses text condition)
TASKS = {
    "depth":   ("depth",  1, 16, False),
    "normal":  ("normal", 3, 16, False),
    "matting": ("mat",    1, 16, False),
    "refseg":  ("refseg", 1, 16, True),
    # saliency (scalar mask) and pose (multi-channel heatmaps + person detection) use the
    # same recipe; their weights are not part of this minimal release. See the paper.
}


class ThinPixelTail(nn.Module):
    """Token (B, N, 128) -> pixel (B, K, H, W). One linear layer + fixed reshape.
    NO convolution, NO upsampling, NO inter-token mixing."""
    def __init__(self, patch_size, out_channels):
        super().__init__()
        self.ps, self.oc = patch_size, out_channels
        self.unpatch_linear = nn.Linear(128, patch_size * patch_size * out_channels)

    def forward(self, tokens, grid_hw):
        B, N, _ = tokens.shape
        Ht, Wt = grid_hw
        x = self.unpatch_linear(tokens).reshape(B, Ht, Wt, self.ps, self.ps, self.oc)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, Ht * self.ps, Wt * self.ps, self.oc)
        return x.permute(0, 3, 1, 2).contiguous()


def fit_resolution(H, W, cap=2048, floor=512):
    """Aspect-preserving (no stretch); long side clamped to [floor, cap]; dims -> multiple of 16."""
    long = max(H, W)
    target = min(max(long, floor), cap)
    s = target / long
    return max(16, round(H * s / 16) * 16), max(16, round(W * s / 16) * 16)


class ReChannel:
    def __init__(self, device="cuda", dtype=torch.bfloat16):
        self.device, self.dtype = device, dtype
        pipe = Flux2KleinPipeline.from_pretrained(snapshot_download(KLEIN_REPO), torch_dtype=dtype)
        self.body = pipe.transformer.to(device).eval()
        self.vae = pipe.vae.to(device).eval()
        self.text_encoder = pipe.text_encoder.to(device).eval()
        self.tokenizer = pipe.tokenizer
        aux = load_file(hf_hub_download(WEIGHTS_REPO, "auxiliary/aux_4b.safetensors"))
        self.bn_mean = aux["bn_mean"].to(device, dtype)
        self.bn_std = aux["bn_std"].to(device, dtype)
        self.null_emb = aux["null_emb"].to(device, dtype)
        self.null_ids = aux["null_text_ids"].to(device)
        # FLUX-2 canonical text ids for T=32: row i = [0, 0, 0, i]
        self.txt_ids_T32 = torch.tensor(
            np.stack([np.zeros(32), np.zeros(32), np.zeros(32), np.arange(32)], -1)[None],
            dtype=torch.float32, device=device)
        self._keys = set(self.body.state_dict().keys())
        self._loaded = {}                               # task -> (lora dict, tail)

    def _load_task(self, task):
        if task in self._loaded:
            return self._loaded[task]
        sub, K, ps, _ = TASKS[task]
        path = hf_hub_download(WEIGHTS_REPO, f"4b/{sub}/lora_tail.safetensors")
        sd = load_file(path)
        lora = {}
        for layer in sorted({k.rsplit('.', 2)[0] for k in sd if 'lora_A' in k}):
            wk = f"{layer.replace('transformer.', '', 1)}.weight"
            if wk not in self._keys:
                continue
            A = sd[f"{layer}.lora_A.weight"].to(self.device).float()   # (rank, in)
            B = sd[f"{layer}.lora_B.weight"].to(self.device).float()   # (out, rank)
            lora[wk] = (A, B)
        tail = ThinPixelTail(ps, K).to(self.device, self.dtype).eval()
        tail.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("tail.")}, strict=True)
        self._loaded[task] = (lora, tail)
        return self._loaded[task]

    def _tokens(self, rin):
        rt = torch.from_numpy(rin.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None].to(self.device, self.dtype)
        z = self.vae.encode(rt).latent_dist.mode().permute(0, 2, 3, 1)
        B_, Hh, Wh, C = z.shape
        Hp, Wp = Hh // 2, Wh // 2
        z = z.reshape(B_, Hp, 2, Wp, 2, C).permute(0, 1, 3, 5, 2, 4).reshape(B_, Hp, Wp, C * 4)
        seq = ((z - self.bn_mean.reshape(1, 1, 1, -1)) / self.bn_std.reshape(1, 1, 1, -1)).reshape(1, -1, 128)
        h = torch.arange(Hp, device=self.device).repeat_interleave(Wp)
        w = torch.arange(Wp, device=self.device).repeat(Hp)
        img_ids = torch.stack([torch.zeros_like(h), h, w, torch.zeros_like(h)], -1).float()
        return seq, img_ids, Hp, Wp

    def _embed_text(self, phrase):
        from torch.nn.attention import sdpa_kernel, SDPBackend
        torch.backends.cuda.enable_cudnn_sdp(False)
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": phrase}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        enc = self.tokenizer([rendered], return_tensors="pt", padding="max_length", truncation=True, max_length=32)
        ids, att = enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
        with sdpa_kernel(SDPBackend.MATH):
            o = self.text_encoder(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False, return_dict=True)
        return torch.cat([o.hidden_states[i] for i in (9, 18, 27)], -1).to(self.dtype)

    @torch.no_grad()
    def _forward(self, rin, te, ti, tail):
        seq, img_ids, Hp, Wp = self._tokens(rin)
        t = torch.full((1,), 0.999, device=self.device, dtype=torch.float32)
        g = torch.zeros(1, device=self.device, dtype=torch.float32)
        while ti.dim() > 2: ti = ti.squeeze(0)
        while te.dim() > 3: te = te.squeeze(0)
        out = self.body(hidden_states=seq, encoder_hidden_states=te, timestep=t,
                        img_ids=img_ids, txt_ids=ti, guidance=g, return_dict=False)[0]
        return tail(out, (Hp, Wp))[0].cpu().float().numpy()

    @torch.no_grad()
    def run(self, image, task, phrase="the object"):
        """image: HxWx3 uint8 RGB. Returns the raw decoded field (task-native)."""
        _, K, ps, use_text = TASKS[task]
        lora, tail = self._load_task(task)
        sd = self.body.state_dict()
        # snapshot ONLY the LoRA-affected layers, merge in-place (float then cast), restore after
        saved = {wk: sd[wk].detach().clone() for wk in lora}
        with torch.no_grad():
            for wk, (A, B) in lora.items():
                sd[wk].copy_((saved[wk].float() + LORA_SCALE * (B @ A)).to(self.dtype))
        try:
            if use_text:                                    # refseg: 512^2 square + text condition
                rin = np.asarray(Image.fromarray(image).resize((512, 512), Image.BILINEAR))
                p = self._forward(rin, self._embed_text(phrase), self.txt_ids_T32, tail)
            else:                                           # depth/normal/matting/saliency: aspect-fit + hflip TTA
                H0, W0 = image.shape[:2]
                Hn, Wn = fit_resolution(H0, W0)
                rin = np.asarray(Image.fromarray(image).resize((Wn, Hn), Image.BILINEAR))
                p = self._forward(rin, self.null_emb, self.null_ids, tail)
                pf = self._forward(rin[:, ::-1].copy(), self.null_emb, self.null_ids, tail)[:, :, ::-1]
                if task == "normal":
                    pf[0] = -pf[0]                          # x-channel negates under horizontal flip
                p = (p + pf) / 2.0
        finally:
            with torch.no_grad():
                for wk, orig in saved.items():
                    sd[wk].copy_(orig)                      # restore frozen backbone
        return decode(task, p)


def decode(task, p):
    """Raw head output -> task-native field."""
    if task == "depth":
        return np.exp(p[0])                                 # scalar depth
    if task == "normal":
        n = p / (np.sqrt((p ** 2).sum(0, keepdims=True)) + 1e-8)
        return n.transpose(1, 2, 0)                         # HxWx3 unit vectors
    if task == "matting":
        return np.clip((np.tanh(p[0]) + 1) / 2, 0, 1)       # alpha in [0,1]
    if task == "saliency":
        return 1.0 / (1.0 + np.exp(-p[0]))                  # sigmoid prob
    if task == "refseg":
        return (p[0] > 0).astype(np.float32)                # binary mask
    raise ValueError(task)


# ---------------------------------------------------------------- visualization
def colorize(task, field, rgb):
    import matplotlib
    if task == "depth":
        d = (field - field.min()) / (field.max() - field.min() + 1e-8)
        return (matplotlib.colormaps["turbo"](1 - d)[..., :3] * 255).astype(np.uint8)
    if task == "normal":
        return ((field * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    if task in ("matting", "saliency"):
        return np.repeat((np.clip(field, 0, 1) * 255).astype(np.uint8)[..., None], 3, -1)
    if task == "refseg":
        H, W = rgb.shape[:2]
        m = np.asarray(Image.fromarray((field * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)) > 127
        ov = rgb.astype(np.float32).copy(); ov[m] = 0.45 * ov[m] + 0.55 * np.array([0, 200, 255])
        return ov.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--tasks", default="depth,normal,matting,refseg")
    ap.add_argument("--phrase", default="the object", help="referring expression for refseg")
    ap.add_argument("--out", default="rechannel_out.png")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    H0, W0 = rgb.shape[:2]
    model = ReChannel(device=args.device)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    panels = [("Input", rgb)]
    for task in tasks:
        field = model.run(rgb, task, phrase=args.phrase)
        vis = colorize(task, field, rgb)
        if vis.shape[:2] != (H0, W0):
            vis = np.asarray(Image.fromarray(vis).resize((W0, H0), Image.LANCZOS))
        label = f'RefSeg "{args.phrase}"' if task == "refseg" else task.capitalize()
        panels.append((label, vis))

    # stitch a labeled row
    from PIL import ImageDraw, ImageFont
    Hh = 360
    ims = [Image.fromarray(a).resize((round(a.shape[1] * Hh / a.shape[0]), Hh), Image.LANCZOS) for _, a in panels]
    LAB, GAP = 42, 10
    Wt = sum(im.size[0] for im in ims) + GAP * (len(ims) - 1)
    canvas = Image.new("RGB", (Wt, Hh + LAB), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    x = 0
    for (name, _), im in zip(panels, ims):
        canvas.paste(im, (x, LAB))
        tb = dr.textbbox((0, 0), name, font=font)
        dr.text((x + (im.size[0] - (tb[2] - tb[0])) // 2, 9), name, fill=(20, 20, 30), font=font)
        x += im.size[0] + GAP
    canvas.save(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
