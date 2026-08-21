#!/bin/bash
# =============================================================================
# Foolproof reconstruction of the environment in which CDI + DiT-RF works.
#
# Target host specs recorded at snapshot time (see system_info.txt):
#   OS        : Ubuntu 24.04.3 LTS, kernel 6.8, glibc 2.39
#   CPU       : Intel Xeon Gold 6526Y (x86_64, 64 CPUs, 503 GiB RAM)
#   GPU       : 3x NVIDIA RTX PRO 5000 Blackwell (sm_12.0, 49 GB each, driver 580.105.08)
#   Python    : 3.12.3  (CPython, Ubuntu 24.04 system python)
#   PyTorch   : 2.10.0+cu128  (cuDNN 9.10.02)
#   flash-attn: 2.8.3+cu128torch2.10  (Tri Dao, prebuilt wheel for torch2.10/cu128/cp312)
#   venv path : duci-pul/.venv
#
# What this script does (idempotent, safe to re-run):
#   1. Verifies host prerequisites (python 3.12, nvidia driver >= 580).
#   2. Creates a fresh venv at duci-pul/.venv  (refuses to overwrite if one exists unless FORCE=1).
#   3. Installs PyTorch 2.10.0+cu128 from the official pytorch.org index.
#   4. Installs the exact flash-attn prebuilt wheel that matches torch 2.10 / cu128 / cp312.
#   5. Installs the rest of the pinned pip-freeze dump (`pip_freeze.txt`).
#   6. Runs a smoke test that imports torch, flash_attn, and loads the DiT-RF wrapper.
#
# Usage:
#   bash cdi/env_snapshot/reconstruct.sh            # will refuse if .venv already exists
#   FORCE=1 bash cdi/env_snapshot/reconstruct.sh    # removes and recreates the venv
#   TARGET_ARCH=blackwell bash ...                  # future-proof: switch wheel URL for new hw
# =============================================================================

set -euo pipefail

# --- paths -------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV="$REPO_ROOT/.venv"
FREEZE="$SCRIPT_DIR/pip_freeze.txt"

# --- pinned versions ---------------------------------------------------------
PY_MAJMIN="3.12"
TORCH_VER="2.10.0"
TORCH_CHANNEL="cu128"                        # CUDA 12.8 wheels
PYTORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CHANNEL}"

# Prebuilt flash-attn wheel — must match torch ABI and GPU arch.
# For Blackwell (sm_12.0) + torch 2.10 + cu128 + python 3.12 on Linux x86_64:
FLASH_WHEEL="${FLASH_WHEEL:-https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.12/flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl}"

# --- helpers -----------------------------------------------------------------
log()  { printf '\033[1;34m[reconstruct]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[reconstruct][WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[reconstruct][ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

# --- preflight ---------------------------------------------------------------
log "Preflight checks…"

command -v python${PY_MAJMIN} >/dev/null 2>&1 \
    || die "python${PY_MAJMIN} not found on PATH. Install it first (Ubuntu: apt install python${PY_MAJMIN} python${PY_MAJMIN}-venv)."
PY_SYS="$(command -v python${PY_MAJMIN})"
PY_SYS_VER="$($PY_SYS -c 'import sys; print("{}.{}.{}".format(*sys.version_info[:3]))')"
log "system python: $PY_SYS ($PY_SYS_VER)"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-smi not found. CUDA smoke test will fail but pip install will still proceed."
else
    DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    log "NVIDIA driver: $DRIVER (need >= 580 for Blackwell; older drivers may require a different torch wheel)"
fi

[ -f "$FREEZE" ] || die "pip_freeze.txt not found at $FREEZE"

# --- venv --------------------------------------------------------------------
if [ -d "$VENV" ]; then
    if [ "${FORCE:-0}" = "1" ]; then
        log "FORCE=1 set — removing existing venv at $VENV"
        rm -rf "$VENV"
    else
        die "Venv already exists at $VENV. Re-run with FORCE=1 to wipe and recreate."
    fi
fi

log "Creating venv at $VENV"
"$PY_SYS" -m venv "$VENV"
PY="$VENV/bin/python3"

log "Upgrading pip/setuptools/wheel inside venv"
"$PY" -m pip install --upgrade pip setuptools wheel

# --- install PyTorch first (matches cu128 ABI expected by flash-attn wheel) --
log "Installing torch==${TORCH_VER} torchvision from ${PYTORCH_INDEX}"
"$PY" -m pip install --index-url "$PYTORCH_INDEX" \
    "torch==${TORCH_VER}" torchvision

# --- flash-attn prebuilt wheel -----------------------------------------------
log "Installing prebuilt flash_attn wheel:"
log "  $FLASH_WHEEL"
"$PY" -m pip install "$FLASH_WHEEL"

# --- everything else (pinned) ------------------------------------------------
# We install from the full pip freeze, but EXCLUDE torch/torchvision/flash_attn
# lines that were already handled above (and would try to reinstall from a
# different index).
log "Installing the rest of the pinned dependencies from pip_freeze.txt"
TMP_FREEZE="$(mktemp)"
grep -v -E '^(torch|torchvision|torchaudio|flash_attn|flash-attn)([ =@]|$)' "$FREEZE" > "$TMP_FREEZE"
"$PY" -m pip install -r "$TMP_FREEZE"
rm -f "$TMP_FREEZE"

# --- smoke test --------------------------------------------------------------
log "Running smoke test…"
"$PY" - <<'PY'
import importlib, sys
import torch
print(f"[smoke] torch               = {torch.__version__}")
print(f"[smoke] torch.version.cuda  = {torch.version.cuda}")
print(f"[smoke] cudnn               = {torch.backends.cudnn.version()}")
print(f"[smoke] gpu count           = {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"[smoke] gpu 0               = {torch.cuda.get_device_name(0)} sm={torch.cuda.get_device_capability(0)}")

import flash_attn
print(f"[smoke] flash_attn          = {flash_attn.__version__}")

# A tiny flash-attn fwd pass in fp16 to ensure the kernel actually dispatches
if torch.cuda.is_available():
    from flash_attn import flash_attn_func
    q = torch.randn(1, 16, 4, 64, device='cuda', dtype=torch.float16)
    k = torch.randn(1, 16, 4, 64, device='cuda', dtype=torch.float16)
    v = torch.randn(1, 16, 4, 64, device='cuda', dtype=torch.float16)
    out = flash_attn_func(q, k, v)
    assert out.shape == q.shape, out.shape
    print("[smoke] flash_attn_func fp16 OK")

# Import CDI wrapper (doesn't instantiate / download any checkpoint)
sys.path.insert(0, 'cdi')
sys.path.insert(0, 'cdi/DiT_MoE')
from src.models.DiT_RFWrapper import DiT_RFWrapper  # noqa: F401
print("[smoke] DiT_RFWrapper import OK")
print("[smoke] all good.")
PY

log "Environment ready at $VENV"
log "Activate with:  source $VENV/bin/activate"
