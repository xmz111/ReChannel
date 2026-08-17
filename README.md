<div align="center">

# ReChannel

### From RGB Generation to Dense Field Readout

### Pixel-Space Dense Prediction with Text-to-Image Models

[![arXiv](https://img.shields.io/badge/arXiv-2607.06553-b31b1b.svg)](https://arxiv.org/abs/2607.06553)
[![Models](https://img.shields.io/badge/🤗%20Models-Hugging%20Face-FFD21E)](https://huggingface.co/xmz111/ReChannel)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**A pretrained text-to-image DiT is already a dense spatial field.
ReChannel simply learns how to read it out.**

</div>

<p align="center">
  <img src="assets/demo.png" width="100%">
</p>

<p align="center">
  <em>
    One image → depth, surface normals, matting, and referring segmentation.<br>
    No convolutional decoder. No upsampling module. No target-side VAE decoding.
  </em>
</p>

---

## ✨ Overview

**ReChannel** turns a pretrained **FLUX-Klein text-to-image DiT** into a general dense predictor with only lightweight task adaptation.

A pretrained DiT already organizes an input image into a **patch-aligned spatial token field**. Instead of attaching a heavy dense prediction decoder, we treat every token as a spatial carrier and simply **re-channel its feature dimensions from RGB appearance to task-native quantities**.

For each task, ReChannel adds only:

* a lightweight **LoRA adapter** on the otherwise frozen DiT backbone;
* a **token-local linear readout head**;
* **no convolution**;
* **no spatial upsampling decoder**;
* **no target-side VAE decoder**.

For scalar outputs, the readout head contains only **~33K parameters**.

> **Core idea:** dense prediction does not require decoding an image.
> Adapt the pretrained token field, then directly read out the desired dense field.

---

## 🔥 News

* **[2026-08]** Native PyTorch 4B training code and task configurations released.
* **[2026-07]** Paper, model weights, and PyTorch inference demo released.

---

## 🎯 Supported Tasks

| Task                       |         Output        |  Demo | Training |
| :------------------------- | :-------------------: | :---: | :------: |
| **Depth**                  | 1-channel dense field |   ✅   |     ✅    |
| **Surface Normal**         | 3-channel dense field |   ✅   |     ✅    |
| **Matting**                |      Alpha matte      |   ✅   |     ✅    |
| **Referring Segmentation** |      Binary mask      |   ✅   |     ✅    |
| **Saliency**               |      Saliency map     | Paper |   Paper  |
| **Human Pose**             |   Keypoint heatmaps   | Paper |   Paper  |

The same token-local readout principle is used across all tasks.

---

## 🚀 Getting Started

### Installation

```bash
pip install -r requirements.txt
```

A CUDA GPU is required.

The following models are downloaded automatically from the Hugging Face Hub on first use:

* Backbone: [`black-forest-labs/FLUX.2-klein-base-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)
* ReChannel task weights: [`xmz111/ReChannel`](https://huggingface.co/xmz111/ReChannel)

You may need to accept the FLUX-Klein license and log in first:

```bash
huggingface-cli login
```

---

## ⚡ Quick Start

Run multiple dense prediction tasks on a single image:

```bash
python infer.py \
  --image assets/demo_input.jpg \
  --tasks depth,normal,matting,refseg \
  --phrase "the right couch" \
  --out out.png
```

### Arguments

* `--tasks`
  Any subset of `depth`, `normal`, `matting`, and `refseg`.

* `--phrase`
  Referring expression used by `refseg`.

* `--no-tta`
  Disable horizontal-flip test-time augmentation.

### Inference Resolution

**Depth / Normal / Matting**

* preserve the native image aspect ratio;
* clamp the long side to **512–2048 px**;
* use horizontal-flip TTA by default;
* average two forward passes.

For a strict single forward pass:

```bash
python infer.py \
  --image assets/demo_input.jpg \
  --tasks depth,normal,matting \
  --no-tta \
  --out out.png
```

**Referring Segmentation**

* runs at **512 × 512**, matching its training resolution;
* uses a single forward pass.

---

## 🧠 How ReChannel Works

```text
                           task LoRA Δt
                                │
                                ▼
RGB ──► VAE Encoder ──► FLUX-Klein DiT ──► token field Z_t
                         frozen backbone          │
                                                  │
                                      token-local linear readout
                                                  │
                                                  ▼
                                reshape(W_t z_ij + b_t)
                                                  │
                                                  ▼
                                         dense output Ŷ
```

For every spatial token (z_{ij}), the task-specific readout is

```text
ŷ_ij = reshape(W_t z_ij + b_t) ∈ R^(p × p × K_t)
```

The predicted patches are tiled back over the image plane to form the final dense field.

The readout head performs **no spatial mixing**. All spatial reasoning is therefore carried by the adapted DiT token field itself rather than by a downstream decoder.

### What is trained?

```text
FLUX-Klein backbone     frozen
Task LoRA Δt            trainable
Linear readout Wt, bt   trainable
```

That's it.

---

## 🏋️ Training

Native PyTorch training is provided for:

* depth;
* surface normals;
* matting;
* referring segmentation.

### Single GPU

```bash
python train.py \
  --task normal \
  --data-root /path/to/data
```

### Multi-GPU

```bash
torchrun --nproc_per_node=2 train.py \
  --task normal \
  --data-root /path/to/data
```

Training datasets are **not distributed with this repository**.

See [`training/train.py`](training/train.py) for the expected local dataset layout and task configuration.

---

## 📐 Lightweight Dense Readout

For token dimension 128 and patch size (p), the head contains

```text
p² × K_t × 128
```

output weights, where (K_t) is the number of task channels.

| Task Type      | (K_t) | Readout Size |
| :------------- | ----: | -----------: |
| Scalar field   |     1 |         ~33K |
| Surface normal |     3 |         ~99K |

Despite this extremely small readout, the model can recover detailed spatial predictions because the dense structure is already represented in the adapted DiT token field.

---

## 📌 Notes

* This repository contains the **4B inference demo** and compact **native PyTorch training code**.
* Benchmark evaluation scripts and datasets are not included.
* Pose requires multi-channel keypoint heatmaps together with person detection and is not included in this minimal demo.
* Saliency and pose follow the same ReChannel principle; see the paper for the complete experimental setup.

---

## 📄 Paper

**From RGB Generation to Dense Field Readout: Pixel-Space Dense Prediction with Text-to-Image Models**

Zanyi Wang, Xin Lin, Haodong Li, Dengyang Jiang, Yijiang Li

📄 [arXiv:2607.06553](https://arxiv.org/abs/2607.06553)
🤗 [Model Weights](https://huggingface.co/xmz111/ReChannel)

---

## 🙏 Acknowledgements

We thank Google's **TPU Research Cloud (TRC)** program for granting us access to Cloud TPUs.

---

## 📜 License

Code in this repository is released under the [MIT License](LICENSE).

The FLUX-Klein backbone and released model weights are subject to their respective licenses. Please review the corresponding licenses before use.

---

## 📝 Citation

If you find ReChannel useful for your research, please consider citing:

```bibtex
@article{wang2026rechannel,
  title   = {From RGB Generation to Dense Field Readout: Pixel-Space Dense Prediction with Text-to-Image Models},
  author  = {Wang, Zanyi and Lin, Xin and Li, Haodong and Jiang, Dengyang and Li, Yijiang},
  journal = {arXiv preprint arXiv:2607.06553},
  year    = {2026}
}
```

<div align="center">

**If you find this project useful, a ⭐ is greatly appreciated!**

</div>
