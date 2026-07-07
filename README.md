# ReChannel

**From RGB Generation to Dense Field Readout: Pixel-Space Dense Prediction with Text-to-Image Models**

ReChannel reads dense prediction targets out of a FLUX-Klein text-to-image DiT
with lightweight per-task LoRA adapters on the (otherwise frozen) backbone.
A pretrained DiT already organizes an RGB image into a patch-aligned spatial token
field; each token is a spatial carrier whose channels we simply **re-channel** from
RGB appearance to task-native quantities with a **token-local linear head** (~33K
params for scalar tasks, one `nn.Linear`, no convolution / no upsampling / no
target-side VAE decoder). Only the task LoRA and the linear head are trained.

![demo](assets/demo.png)

> One input image → depth, surface normal, matting, and referring segmentation
> (`"the right couch"`), all read out from the same token field by a token-local linear
> head. (Saliency and pose use the same recipe and are shown in the paper.)

## Install

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU. The frozen backbone `black-forest-labs/FLUX.2-klein-base-4B`
(you may need to accept its license and `huggingface-cli login`) and the per-task
LoRA + linear heads (`xmz111/ReChannel`) are downloaded automatically
from the Hugging Face Hub on first run.

## Quick start (single image, all tasks)

```bash
python infer.py --image assets/demo_input.jpg \
    --tasks depth,normal,matting,refseg \
    --phrase "the right couch" \
    --out out.png
```

- `--tasks`: any subset of `depth, normal, matting, refseg`.
- `--phrase`: the referring expression used by `refseg` (text-conditioned).
- depth / normal / matting run at aspect-preserving native resolution
  (long side clamped to 512–2048, no stretch) with horizontal-flip TTA; refseg
  runs at 512² (its training resolution).

## How it works (per task)

```
RGB --VAE encoder--> latent tokens --DiT (frozen θ + task LoRA Δt, σ=0)--> token field Z_t
    --  Ŷ = reshape( W_t · z_ij + b_t ) ∈ R^{p×p×K}  --tile over the plane-->  dense field
```

The backbone is frozen; only a lightweight per-task LoRA adapter and the linear
head are trained. The head has no spatial mixing — all spatial structure comes
from the adapted token field, not the head.

## Notes

- This repository is an **inference / qualitative-demo** release. It is not the
  benchmark-evaluation pipeline used to produce the paper's tables.
- Head size is `p²·Kₜ × 128`: 33K for scalar tasks (K=1), 99K for surface normals
  (K=3).
- Pose (multi-channel keypoint heatmaps + person detection) is not included in
  this minimal demo; see the paper for the full recipe.

## License

Code in this repository is released under the MIT License (see `LICENSE`).
The FLUX-Klein backbone and the released weights are subject to their own
licenses; please review them before use.

## Citation

```bibtex
@article{wang2026rechannel,
  title   = {From RGB Generation to Dense Field Readout: Pixel-Space Dense Prediction with Text-to-Image Models},
  author  = {Wang, Zanyi and Lin, Xin and Li, Haodong and Jiang, Dengyang and Li, Yijiang and Xie, Pengtao},
  journal = {arXiv preprint},
  year    = {2026}
}
```
