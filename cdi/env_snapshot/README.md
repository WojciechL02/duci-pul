# CDI + DiT-RF Environment Snapshot

Frozen capture of the working environment for `cdi` + DiT-RF feature extraction, so the same setup can be recreated months later (or on a different machine) without re-discovering all the versions / build trivia.

**Snapshot date:** 17 Apr 2026
**Repo HEAD at snapshot:** `4e2e70d` on branch `full-pipeline`
**Git remote:** `https://github.com/WojciechL02/duci-pul.git`

---

## Why this exists

The DiT-RF pipeline has three brittle integration points:

1. **GPU arch ↔ flash-attn ABI.** The machine uses Blackwell (sm_12.0) GPUs. `flash-attn` has to be built against the exact torch + CUDA ABI it will run on; source builds on Blackwell are hours-long and historically failed (sm_120 backward-pass segfault). The fix is the community **prebuilt wheel** from [mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels).
2. **Mandatory flash-attn.** The CDI DiT wrapper refuses to construct if `flash_attn` isn't importable — numerical results depend on the flash-attn MHA layout (`Wqkv`, `out_proj`, `q_norm`, `k_norm`). Using the non-flash fallback gives silently wrong features.
3. **Torch wheel index.** For Blackwell you specifically need the `cu128` wheels from `download.pytorch.org`; `pip install torch` without `--index-url` picks up the wrong ABI.

Once those three lock in, the rest of the stack (diffusers, hydra, hf-hub, etc.) is a straightforward `pip install -r`.

---

## Files in this directory

| file | purpose |
|---|---|
| `system_info.txt`  | OS, kernel, CPU, RAM, disk, GPU, driver, nvcc, python, torch, flash_attn, repo git state, venv layout at snapshot time. |
| `pip_freeze.txt`   | Full `pip freeze --all` of the working `.venv` — every direct and transitive dep pinned to an exact version. Includes the flash-attn wheel URL with sha256. |
| `pip_show_key.txt` | `pip show` output for the key packages (torch, flash_attn, hydra, diffusers, transformers, numpy, scipy, omegaconf, etc.) — installed paths, requires / required-by graph. |
| `pyvenv.cfg`       | Copy of `.venv/pyvenv.cfg` — records the system python interpreter, version, and the venv-creation command. |
| `reconstruct.sh`   | Idempotent reconstruction script (see below). |

All files are plain text and safe to commit.

---

## Pinned core versions

| component | version |
|---|---|
| OS                | Ubuntu 24.04.3 LTS (kernel 6.8.0-101, glibc 2.39) |
| Python            | **3.12.3** (system CPython) |
| PyTorch           | **2.10.0+cu128** |
| torch.backends.cudnn | 9.10.02 |
| flash-attn        | **2.8.3+cu128torch2.10** (prebuilt wheel, cp312, linux_x86_64) |
| CUDA runtime (wheel) | 12.8 |
| NVIDIA driver     | 580.105.08 |
| GPU               | 3 × NVIDIA RTX PRO 5000 Blackwell (sm 12.0, 49 GB each) |
| diffusers         | 0.36.0 |
| transformers      | 4.57.6 |
| hydra-core        | 1.3.2 |
| omegaconf         | 2.3.0 |
| numpy / scipy     | 2.3.5 / 1.17.0 |

Exact install lines:

```bash
# PyTorch (cu128 ABI)
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision

# flash-attn prebuilt wheel (Blackwell-compatible, matches torch 2.10 + cu128 + cp312)
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.12/flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl
```

The second URL is the single most important line in this snapshot — without it you either spend hours building flash-attn from source on Blackwell or end up with a torch-ABI-mismatch `undefined symbol` error at import time.

---

## Reconstructing the environment

On a compatible host (Blackwell + driver ≥ 580 + Python 3.12), from the repo root:

```bash
bash cdi/env_snapshot/reconstruct.sh
```

This is idempotent; it:

1. Verifies `python3.12` and `nvidia-smi` are present.
2. Creates a fresh `.venv` at `duci-pul/.venv` (refuses to overwrite; pass `FORCE=1` to wipe).
3. Installs `torch==2.10.0+cu128` from the pytorch.org index.
4. Installs the exact flash-attn prebuilt wheel.
5. Installs the rest of `pip_freeze.txt` (minus torch/flash_attn, which were already handled).
6. Runs a smoke test: imports `torch`, allocates a flash-attn fp16 forward pass on GPU 0, imports `DiT_RFWrapper`.

Smoke-test success means DiT-RF feature extraction will run identically to when the snapshot was taken. After that you can:

```bash
source .venv/bin/activate
bash cdi/bash_scripts/run_synthetic_cdi.sh   # full pipeline, all models, all datasets
```

---

## Migrating to a different machine

If you move off this host, only three things can realistically change and still keep DiT-RF working:

1. **Different GPU arch.** If the new box has Ampere (sm_80/86), Hopper (sm_90), or Ada (sm_89), you'll need a different flash-attn wheel. Check the same [mjun0812 release page](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases) for a wheel matching your torch version + cuda tag + python + GPU. Override via env var:

   ```bash
   FLASH_WHEEL="https://.../flash_attn-…-cp312-cp312-linux_x86_64.whl" \
   bash cdi/env_snapshot/reconstruct.sh
   ```

2. **Different Python.** The prebuilt wheel is pinned to `cp312` — on Python 3.10 or 3.11 you need the corresponding `cp310`/`cp311` wheel from the same release.

3. **Different torch minor.** flash-attn wheels are ABI-locked to a specific torch build (2.10 in our case). If you bump torch you must bump the flash-attn wheel to match. Do **not** mix torch 2.11 with a flash-attn wheel built against torch 2.10 — you'll get an `undefined symbol` error at `import flash_attn`.

`pip_freeze.txt` pins every other package; the rest of the stack is version-stable.

---

## Known gotchas preserved by this snapshot

- **Never disable flash attention.** `DiT_RFWrapper.py` and `DiT_MoE/models.py` both raise if `use_flash_attn=false` or `flash_attn` is missing. This is intentional — the model was trained with flash-attn MHA, and the non-flash fallback produces numerically different attention, which silently corrupts CLiD / CarliniLT / gradient_masking features.
- **Checkpoint remap is direction-sensitive.** `_remap_checkpoint_keys` renames legacy `attn.qkv` / `attn.proj` → `attn.Wqkv` / `attn.out_proj`. Reversing this (which an old version did) causes `load_state_dict(strict=False)` to silently leave the attention weights random-initialized. XL and G checkpoints already ship with the `Wqkv`/`out_proj` names; the B checkpoint ships with legacy names and needs the rename.
- **VAE `.half()` is tied to `model_dtype=float16`.** `_configure_half_precision_hooks` in the wrapper casts embedders' outputs back to `model_dtype`. If you change `model_dtype` to bfloat16 or float32 you need to revisit that path.
