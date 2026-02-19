# NU-DUI: Dataset Usage Inference without Shadow Models or Held-out Data

Code for the paper *Dataset Usage Inference without Shadow Models or Held-out Data*.

Existing Dataset Usage Inference (DUI) methods require training expensive shadow models and assume access to a trusted held-out set known to be absent from training -- assumptions that rarely hold for modern large-scale generative models. **NU-DUI** removes both constraints. It generates synthetic non-member samples, extracts diverse membership signals, and recasts DUI as a negative-unlabeled (NU) learning problem to estimate what fraction of a candidate dataset was used during training.

The pipeline has three stages, each corresponding to a directory in this repo:

- **`synthetic/`** -- Generate synthetic non-member samples by running real images through autoregressive visual models (VAR, Infinity, RAR) at controllable generation levels, producing paired member/non-member datasets.
- **`cdi/`** -- Extract membership-inference features from the target model (MIA signals such as loss curvature, gradient masking, CLiD, etc.), compute per-sample scores, and aggregate them.
- **`src/`** -- Perform the NU-learning step: given the MIA scores, estimate the proportion of the candidate dataset that was used in training via positive-unlabeled learning methods with post-hoc correction.

## Repository Layout

```
duci-pul/
├── synthetic/                  # Synthetic dataset generation
│   ├── create_dataset_var.py
│   ├── create_dataset_infinity.py
│   ├── create_dataset_rar.py
│   ├── download_models.sh      # Download all generator weights
│   ├── VAR/                    # VAR model code
│   ├── Infinity/               # Infinity model code
│   ├── RAR/                    # RAR model code
│   ├── weights/                # Model checkpoints (after download)
│   └── datasets/               # Generated datasets land here
├── cdi/                        # CDI membership inference pipeline
│   ├── main.py                 # Hydra-based entry point
│   ├── conf/                   # Hydra configs (model, attack, dataset, action)
│   ├── src/                    # Models, dataloaders, attacks, evaluation
│   ├── bash_scripts/           # Orchestration scripts
│   ├── out/                    # Output features, scores, results
│   └── requirements.txt
├── src/                        # PU-learning methods
│   ├── test_pul_methods.py
│   ├── PUBiasCalibration/
│   └── run.sh
├── requirements.txt
├── environment.yml
└── README.md                   # This file
```

## Environment Setup

### 1. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install PyTorch

**For NVIDIA Blackwell GPUs (RTX 50 series)** -- requires PyTorch nightly with CUDA 12.8:
```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

**For older NVIDIA GPUs (Ampere, Hopper, etc.)** -- stable PyTorch with CUDA 12.4:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r cdi/requirements.txt
```

### 4. Clone External Dependencies

The CDI pipeline depends on three external codebases:
```bash
cd cdi
git clone --depth 1 https://github.com/baofff/U-ViT.git U-ViT
git clone --depth 1 https://github.com/feizc/DiT-MoE.git DiT_MoE
git clone --depth 1 https://github.com/CompVis/latent-diffusion.git latent-diffusion
cd ..
```

### 5. Verify Installation

```bash
python cdi/smoke_imports.py
```

---

## Downloading Source Data

### ImageNet

ImageNet must be in the standard ImageFolder format with `train/` and `val/` directories each containing 1000 class folders (e.g. `n01440764/`).

**Option 1: Academic Torrents (recommended for full dataset)**
```bash
pip install libtorrent
python cdi/helper_scripts/download_imagenet_torrent.py --split train --output /path/to/imagenet
python cdi/helper_scripts/download_imagenet_torrent.py --split val --output /path/to/imagenet
bash cdi/helper_scripts/extract_imagenet.sh /path/to/imagenet
```

**Creating a subset (5 images per class = 5000 images):**
```bash
python cdi/helper_scripts/create_imagenet_subset.py \
    --src /path/to/imagenet \
    --dst cdi/data/imagenet_5000 \
    --images-per-class 5
```

### MS-COCO (for UViT text-to-image models only)

See [cdi/README.md](cdi/README.md) for full COCO setup instructions including text-embedding extraction.

---

## Generating Synthetic Datasets

The `synthetic/` directory contains scripts that take real ImageNet images and produce partially-generated versions using three autoregressive visual models. Each script creates **tp** (true-positive, sourced from ImageNet `train`) and **fp** (false-positive, sourced from ImageNet `val`) dataset pairs.

### Download Generator Weights

```bash
bash synthetic/download_models.sh
```

This downloads checkpoints for all three generators into `synthetic/weights/`.

### VAR (Visual Autoregressive Modeling)

VAR uses 10 hierarchical scales. The `--last_scales` parameter controls how many of the finest scales are model-generated (more scales = more synthetic content).

```bash
cd synthetic
python create_dataset_var.py \
    --imagenet_dir /path/to/imagenet \
    --output_root datasets \
    --last_scales 4 \
    --cfg 1.5 \
    --gpu 0
```

| Parameter | Description | Default |
|---|---|---|
| `--last_scales` | Number of scales to generate (last N of 10) | *required* |
| `--cfg` | Classifier-free guidance scale | 1.5 |
| `--top_k` | Top-k sampling | 900 |
| `--top_p` | Top-p (nucleus) sampling | 0.96 |
| `--max_images` | Max images per class to process | all |
| `--skip_tp` / `--skip_fp` | Skip generating one of the splits | -- |

**Output:** `datasets/imagenet_{tp,fp}_var_last{N}_cfg{C}/`

### Infinity

Infinity uses 7 scales at 256x256. The `--last_scales` parameter controls generation depth.

```bash
cd synthetic
python create_dataset_infinity.py \
    --imagenet_dir /path/to/imagenet \
    --output_root datasets \
    --last_scales 2 \
    --cfg 3.0 \
    --gpu 0
```

| Parameter | Description | Default |
|---|---|---|
| `--last_scales` | Number of scales to generate (last N of 7) | *required* |
| `--cfg` | Classifier-free guidance scale | 3.0 |
| `--tau` | Temperature | -- |
| `--pn` | Resolution token budget (`0.06M`, `0.25M`, `1M`) | `0.06M` |

**Output:** `datasets/imagenet_{tp,fp}_infinity_last{N}_cfg{C}/`

### RAR (Randomized Autoregressive)

RAR tokenizes images into 256 discrete tokens on a 16x16 grid. The `--gen_ratio` controls what fraction of tokens are model-generated.

```bash
cd synthetic
python create_dataset_rar.py \
    --imagenet_dir /path/to/imagenet \
    --output_root datasets \
    --gen_ratio 0.25 \
    --mode suffix \
    --gpu 0
```

| Parameter | Description | Default |
|---|---|---|
| `--gen_ratio` | Fraction of tokens that are model-generated | *required* |
| `--mode` | `suffix` (contiguous block at end) or `random` (per-token) | `suffix` |
| `--guidance_scale` | Classifier-free guidance scale | -- |
| `--temperature` | Sampling temperature | -- |

**Output:** `datasets/imagenet_{tp,fp}_rar_gen{R}_{mode}_cfg{C}/`

### Output Dataset Structure

Each generated dataset follows the ImageFolder convention:

```
datasets/imagenet_tp_var_last4_cfg1.5/
├── train/          # Source images autoencoded through the model's VQVAE/tokenizer
│   ├── n01440764/
│   └── ...         (1000 class folders)
├── val/            # img2img: first K scales real, last N scales generated
│   ├── n01440764/
│   └── ...
└── metadata.md     # Generation hyperparameters and timing
```

- **tp** datasets are derived from ImageNet **train** (the model's actual training data).
- **fp** datasets are derived from ImageNet **val** (data the model has not seen during training).
- **train/** contains pure autoencoder reconstructions (no generation).
- **val/** contains img2img outputs where the coarsest scales come from the real image and the finest scales are model-generated.

---

## Running the CDI Pipeline

The CDI pipeline uses [Hydra](https://hydra.cc/) for configuration. All commands run from the `cdi/` directory.

### Downloading CDI Model Checkpoints

**DiT-RF (DiT-MoE):**
```bash
mkdir -p cdi/model_checkpoints/dit_rf
wget -O cdi/model_checkpoints/dit_rf/dit_moe_xl_8E2A.pt \
    "https://huggingface.co/feizhengcong/DiT-MoE/resolve/main/dit_moe_xl_8E2A.pt?download=true"
```

The VAE (`stabilityai/sd-vae-ft-mse`) is automatically downloaded from HuggingFace on first run.

**UViT models:** see [cdi/README.md](cdi/README.md) for download instructions.

### Creating a Dataset Config

For each synthetic dataset, create a YAML file in `cdi/conf/dataset/`. Example for `tp_var_last4`:

```yaml
name: tp_var_last4
dataset_path: /absolute/path/to/synthetic/datasets/imagenet_tp_var_last4_cfg1.5
split: train
test_name: val
train_name: train
```

### Single-Command Usage

```bash
cd cdi
python main.py +model=$MODEL +action=$ACTION +attack=$ATTACK +dataset=$DATASET
```

### Feature Extraction

Extract membership-inference features for a given attack:

```bash
python main.py +model=dit_rf +action=features_extraction +attack=gradient_masking +dataset=tp_var_last4
```

### Score Computation

Compute scores from extracted features:

```bash
python main.py +model=dit_rf +action=scores_computation +attack=carlini_lt +dataset=tp_var_last4
```

### Combination Attack

The `combination_attack` concatenates features and scores from multiple individual attacks (`carlini_lt`, `gradient_masking`, `multiple_loss`, `clid`) and trains a logistic regression classifier. It requires features for all four attacks and scores for `carlini_lt` to already exist.

```bash
# Extract combination features (reads pre-computed individual features/scores)
python main.py +model=dit_rf +action=features_extraction +attack=combination_attack +dataset=tp_var_last4

# Compute combination scores
python main.py +model=dit_rf +action=scores_computation +attack=combination_attack +dataset=tp_var_last4
```

### Batch Pipeline

To run the full pipeline across all synthetic datasets:

```bash
bash cdi/bash_scripts/run_synthetic_cdi.sh
```

This script runs four phases:
1. **GPU Feature Extraction** -- 4 attacks x 16 datasets x 2 splits = 128 jobs, distributed across available GPUs using a lock-based queue.
2. **Carlini Scores (CPU)** -- computes carlini_lt scores for all datasets/splits.
3. **Combination Features (CPU)** -- assembles combination attack features.
4. **Combination Scores (CPU)** -- computes final combination attack scores.

### Available Options

**Models:** `dit_rf`, `uvit_t2i`, `uvit_t2i_deep`

**Actions:** `features_extraction`, `scores_computation`, `evaluation`, `evaluation_bulk`

**Attacks:** `carlini_lt`, `gradient_masking`, `multiple_loss`, `clid`, `combination_attack`, `secmi_stat`, `pia`, `pian`, `noise_optim`, `pandora`

### Evaluation

```bash
python main.py +model=dit_rf +action=evaluation +attack=combination_attack +dataset=tp_var_last4

# Bulk evaluation across all attacks
python main.py +model=dit_rf +action=evaluation_bulk +dataset=tp_var_last4
```

Results are saved to `cdi/out/results/`. Use `organize_results.py` to format them:

```bash
python cdi/organize_results.py cdi/out/results/results.csv
```

### Key Configuration

Edit `cdi/conf/config.yaml` to adjust global parameters:

| Parameter | Description | Default |
|---|---|---|
| `n_samples_eval` | Number of samples used for evaluation | 2500 |
| `train_samples` | Samples for the combination attack's internal classifier training | 2500 |
| `valid_samples` | Validation samples (0 to skip validation) | 0 |
| `run_id` | Identifier embedded in output filenames | `5k` |
| `seed` | Random seed | 0 |

---

## Running PU-Learning Methods

After generating synthetic datasets and obtaining CDI scores, the final step is to run the PU-learning (Positive-Unlabeled Learning) methods to perform dataset-level inference.

### Single Run

```bash
python3 src/test_pul_methods.py -nsym 3 -prob 0.5
```

### All Experiments

```bash
cd src
./run.sh
```

---

For detailed documentation on CDI internals, UViT/COCO setup, and additional options, see [cdi/README.md](cdi/README.md).
