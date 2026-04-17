# Dataset Inference for Diffusion Models

## Environment Setup

> **Need exact reproducibility?** The environment that was verified to work end-to-end (Blackwell GPUs, flash-attn, DiT-RF) is frozen under `cdi/env_snapshot/`. Run `bash cdi/env_snapshot/reconstruct.sh` from the repo root to get a byte-for-byte matching `.venv`. See `cdi/env_snapshot/README.md` for details and migration notes.

### 1. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install PyTorch

**For NVIDIA Blackwell GPUs (RTX 50 series)** - requires PyTorch nightly with CUDA 12.8:
```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

**For older NVIDIA GPUs (Ampere, Hopper, etc.)** - use stable PyTorch with CUDA 12.4:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Initialize Git Submodules

The project requires three submodules (U-ViT, DiT_MoE, latent-diffusion):

```bash
git submodule update --init --recursive
```

Or clone them manually:
```bash
git clone --depth 1 https://github.com/baofff/U-ViT.git U-ViT
git clone --depth 1 https://github.com/feizc/DiT-MoE.git DiT_MoE
git clone --depth 1 https://github.com/CompVis/latent-diffusion.git latent-diffusion
```

## Downloading Models

### DiT-MoE (DiT-RF)

Two DiT-MoE variants are registered as `dit_rf` (DiT-MoE-XL/2) and `dit_rf_g` (DiT-MoE-G/2). Both run as rectified-flow models. The wrapper (`src/models/DiT_RFWrapper.py`) auto-downloads the checkpoint on first use if it is missing on disk (destination `model_checkpoints/dit_rf/`, URL comes from the corresponding model config in `conf/model/`).

| alias      | arch       | params  | checkpoint file         |
|------------|------------|---------|-------------------------|
| `dit_rf`   | `DiT-XL/2` | ≈ 4.2 B | `dit_moe_xl_8E2A.pt`    |
| `dit_rf_g` | `DiT-G/2`  | ≈ 16.5 B| `dit_moe_g_16E2A.pt`    |

Both checkpoints use the flash-attention MHA layout (`Wqkv`, `out_proj`, `q_norm`, `k_norm`). A working `flash_attn` install is **mandatory** — the wrapper and the underlying `DiT_MoE/models.py` block construct if the library is missing or `use_flash_attn=false`. On Blackwell (RTX 50 / sm_120) the community prebuilt wheel works out of the box:

```bash
.venv/bin/python3 -m pip install \
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.12/flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl"
```

If you want to seed the checkpoint directory manually instead of relying on auto-download:

```bash
mkdir -p model_checkpoints/dit_rf
wget -O model_checkpoints/dit_rf/dit_moe_xl_8E2A.pt \
    "https://huggingface.co/feizhengcong/DiT-MoE/resolve/main/dit_moe_xl_8E2A.pt?download=true"
wget -O model_checkpoints/dit_rf/dit_moe_g_16E2A.pt \
    "https://huggingface.co/feizhengcong/DiT-MoE/resolve/main/dit_moe_g_16E2A.pt?download=true"
```

The VAE (`stabilityai/sd-vae-ft-mse`) is automatically downloaded from HuggingFace on first run.

### UViT Models (for COCO text-to-image)

Download the UViT checkpoints for MS-COCO:

```bash
pip install gdown
mkdir -p model_checkpoints/uvit_t2i

# Download UViT-S/2 model
gdown "15JsZWRz2byYNU6K093et5e5Xqd4uwA8S" -O model_checkpoints/uvit_t2i/mscoco_uvit_small.pth

# Download UViT-S/2 Deep model
gdown "1gHRy8sn039Wy-iFL21wH8TiheHK8Ky71" -O model_checkpoints/uvit_t2i/mscoco_uvit_small_deep.pth

# Download autoencoder
gdown "130Cq8uFKEqK8sgroIwN7hnRdNvB9DkCo" -O model_checkpoints/uvit_t2i/autoencoder_kl.pth
```

Or use the helper script:
```bash
bash helper_scripts/download_uvit_checkpoints.sh
```

## Downloading Data

### Path Convention (Relative Paths)

This repo now uses relative paths in configs and helper scripts.

- Run CDI commands from `duci-pul/cdi/`.
- Keep real datasets under `duci-pul/data/` (one level above `cdi/`).
- Use `../data/...` in `conf/dataset/*.yaml` and model configs.

Typical layout:

```
duci-pul/
├── cdi/
└── data/
    ├── imagenet_1k/
    ├── imagenet_5000/
    ├── coco/
    └── coco_5000/
```

### ImageNet

ImageNet must be in the standard ImageFolder format:

```
imagenet/
├── train/
│   ├── n01440764/
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
│   │   └── ...
│   ├── n01443537/
│   │   └── ...
│   └── ... (1000 class folders)
└── val/
    ├── n01440764/
    │   └── ...
    └── ... (1000 class folders)
```

**Option 1: Academic Torrents (recommended for full dataset)**
```bash
pip install libtorrent
python helper_scripts/download_imagenet_torrent.py --split train --output ../data/imagenet_1k
python helper_scripts/download_imagenet_torrent.py --split val --output ../data/imagenet_1k
bash helper_scripts/extract_imagenet.sh ../data/imagenet_1k
```

**Creating a subset (5 images per class → 5000 images, actual files in `data/imagenet_5000`):**
```bash
python helper_scripts/create_imagenet_subset.py \
    --src ../data/imagenet_1k \
    --dst data/imagenet_5000 \
    --images-per-class 5
```
Use `+dataset=imagenet_5000`. Full ImageNet (default config):
```yaml
name: imagenet_real
dataset_path: ../data/imagenet_1k
split: train  # or val
```

### MS-COCO

COCO 2014 is required for UViT text-to-image models.

**Step 1: Download COCO 2014**

```bash
# Using helper script (recommended)
bash helper_scripts/download_coco.sh ../data/coco

# Or manually:
mkdir -p ../data/coco && cd ../data/coco
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip train2014.zip && unzip val2014.zip && unzip annotations_trainval2014.zip
```

**Step 2: Extract text embeddings**

The UViT models require pre-extracted CLIP text embeddings:

```bash
source .venv/bin/activate
python helper_scripts/uvit_extract_embeddings.py --coco-dir ../data/coco --split train
python helper_scripts/uvit_extract_embeddings.py --coco-dir ../data/coco --split val
```

Expected directory structure after setup:
```
coco/
├── train2014/
│   ├── COCO_train2014_000000000001.jpg
│   └── ...
├── val2014/
│   ├── COCO_val2014_000000000001.jpg
│   └── ...
├── annotations/
│   ├── captions_train2014.json
│   └── captions_val2014.json
├── train_text_emb/
│   ├── 0_0.npy, 0_1.npy, 0_2.npy, 0_3.npy
│   └── ... (4 embeddings per image)
└── val_text_emb/
    └── ...
```

**Creating a subset (5000 images, actual files in `data/coco_5000`):**
```bash
python helper_scripts/create_coco_subset.py \
    --src ../data/coco \
    --dst data/coco_5000 \
    --num-train 5000 \
    --num-val 5000
```
Use `+dataset=mscoco_5000` (config points to `data/coco_5000`). Full COCO (default config):
```yaml
name: mscoco_real
dataset_path: ../data/coco
split: train  # or val
```

## Running the Code

### Basic Usage

```bash
python main.py +action=$action +attack=$attack +model=$model +dataset=$dataset
```

### Example: Feature Extraction with DiT-RF on ImageNet

```bash
# XL variant (default)
python main.py +model=dit_rf +action=features_extraction +attack=gradient_masking +dataset=imagenet

# G variant (smaller batch recommended)
python main.py +model=dit_rf_g +action=features_extraction +attack=gradient_masking +dataset=imagenet model.batch_size=4
```

### Full Pipeline: All Models × All Datasets

`bash_scripts/run_synthetic_cdi.sh` orchestrates the complete 4-phase pipeline (GPU features → carlini scores → combination features → combination scores) across every `(MODEL, DATASET, SPLIT)` combination, parallelising GPU work via a lock-based queue. Defaults cover `dit_rf` + `dit_rf_g` over `imagenet_5pc` (real), `tp_var_last4`, `fp_var_last4`. Override via env vars — e.g.:

```bash
MODELS="dit_rf dit_rf_g" \
DATASETS="imagenet_5pc tp_var_last4 fp_var_last4" \
GPUS="0 1 2" \
IMAGENET_PATH=/data/imagenet_1k \
OUT_ROOT=out_combi \
bash bash_scripts/run_synthetic_cdi.sh
```

### Example: Feature Extraction with UViT on COCO

```bash
# UViT-S/2 on COCO train split
python main.py +model=uvit_t2i +action=features_extraction +attack=gradient_masking +dataset=mscoco_real

# UViT-S/2 on COCO val split
python main.py +model=uvit_t2i +action=features_extraction +attack=gradient_masking +dataset=mscoco_real ++dataset.split=val

# UViT-S/2 Deep model
python main.py +model=uvit_t2i_deep +action=features_extraction +attack=gradient_masking +dataset=mscoco_real
```

### Available Options

**Models:**
- `dit_rf` - DiT-MoE-XL/2 (RF) for ImageNet
- `dit_rf_g` - DiT-MoE-G/2 (RF) for ImageNet (≈ 16.5 B params, needs ≈ 30 GB VRAM at bs=4)
- `uvit_t2i` - UViT-S/2 for COCO text-to-image
- `uvit_t2i_deep` - UViT-S/2 Deep for COCO text-to-image

**Actions:** `features_extraction`, `scores_computation`, `evaluation`, `evaluation_bulk`

**Attacks:** `carlini_lt`, `secmi_stat`, `combination_attack`, `pia`, `pian`, `gradient_masking`, `multiple_loss`, `noise_optim`, `pandora`, `clid`

**Datasets:**
- ImageNet: `imagenet`, `imagenet_5000`, `imagenet100`, `imagenet_tp`, `imagenet_fp`, `imagenetv2`
- COCO: `mscoco_real`, `mscoco_5000`

### Dataset Configuration

Each dataset supports split selection via the config:
```bash
# Train split (default)
python main.py ... +dataset=imagenet

# Validation split
python main.py ... +dataset=imagenet ++dataset.split=val
```

### Batch Size and Sample Count

```bash
# Reduce batch size for limited GPU memory
python main.py ... model.batch_size=4

# Limit number of samples for quick testing
python main.py ... cfg.n_samples_eval=100
```
