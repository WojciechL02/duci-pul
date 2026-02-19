# Dataset Inference for Diffusion Models

## Environment Setup

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

### 5. Verify Installation

```bash
python smoke_imports.py
```

## Downloading Models

### DiT-MoE Model (DiT_RF)

```bash
mkdir -p model_checkpoints/dit_rf
wget -O model_checkpoints/dit_rf/dit_moe_xl_8E2A.pt \
    "https://huggingface.co/feizhengcong/DiT-MoE/resolve/main/dit_moe_xl_8E2A.pt?download=true"
```

The VAE model (`stabilityai/sd-vae-ft-mse`) is automatically downloaded from HuggingFace on first run.

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
python helper_scripts/download_imagenet_torrent.py --split train --output /path/to/imagenet
python helper_scripts/download_imagenet_torrent.py --split val --output /path/to/imagenet
bash helper_scripts/extract_imagenet.sh /path/to/imagenet
```

**Creating a subset (5 images per class → 5000 images, actual files in `data/imagenet_5000`):**
```bash
python helper_scripts/create_imagenet_subset.py \
    --src /path/to/imagenet \
    --dst data/imagenet_5000 \
    --images-per-class 5
```
Use `+dataset=imagenet_5000`. Full ImageNet: update `conf/dataset/imagenet.yaml` with your path:
```yaml
name: imagenet_real
dataset_path: /path/to/your/imagenet
split: train  # or val
```

### MS-COCO

COCO 2014 is required for UViT text-to-image models.

**Step 1: Download COCO 2014**

```bash
# Using helper script (recommended)
bash helper_scripts/download_coco.sh /path/to/coco

# Or manually:
mkdir -p /path/to/coco && cd /path/to/coco
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip train2014.zip && unzip val2014.zip && unzip annotations_trainval2014.zip
```

**Step 2: Extract text embeddings**

The UViT models require pre-extracted CLIP text embeddings:

```bash
source .venv/bin/activate
python helper_scripts/uvit_extract_embeddings.py --coco-dir /path/to/coco --split train
python helper_scripts/uvit_extract_embeddings.py --coco-dir /path/to/coco --split val
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
    --src /path/to/coco \
    --dst data/coco_5000 \
    --num-train 5000 \
    --num-val 5000
```
Use `+dataset=mscoco_5000` (config points to `data/coco_5000`). Full COCO: update `conf/dataset/mscoco_real.yaml` with your path:
```yaml
name: mscoco_real
dataset_path: /path/to/your/coco
split: train  # or val
```

## Running the Code

### Basic Usage

```bash
python main.py +action=$action +attack=$attack +model=$model +dataset=$dataset
```

### Example: Feature Extraction with DiT-RF on ImageNet

```bash
python main.py +model=dit_rf +action=features_extraction +attack=gradient_masking +dataset=imagenet
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
- `dit_rf` - DiT-MoE model for ImageNet
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
