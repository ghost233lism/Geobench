#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash ms-swift/data_clean/setup_vlm_env.sh
# Optional env vars:
#   PYTHON_BIN=python3.10
#   PIP_BIN=pip
#   SWIFT_REPO=/nfs/sunboyuan/Geobench/ms-swift
#   INSTALL_GCC=1
#   CAUSAL_CONV_WHL=/abs/path/causal_conv1d-...cp310...whl
#   FLASH_ATTN_WHL=/abs/path/flash_attn-...cp310...whl

PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_BIN="${PIP_BIN:-pip}"
SWIFT_REPO="${SWIFT_REPO:-/nfs/sunboyuan/Geobench/ms-swift}"
INSTALL_GCC="${INSTALL_GCC:-1}"
CAUSAL_CONV_WHL="${CAUSAL_CONV_WHL:-causal_conv1d-1.5.4+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"
FLASH_ATTN_WHL="${FLASH_ATTN_WHL:-flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: command not found: $1" >&2
    exit 1
  }
}

require_cmd "$PYTHON_BIN"
require_cmd "$PIP_BIN"

log "Checking Python version"
PY_VER="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PY_VER" != "3.10" ]]; then
  echo "ERROR: This script expects Python 3.10 because provided local wheels are cp310." >&2
  echo "Current Python: $PY_VER" >&2
  exit 1
fi

log "Upgrading pip tooling"
$PIP_BIN install -U pip setuptools wheel

log "1) Installing PyTorch CUDA 12.4"
$PIP_BIN install \
  torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu124

log "2) Installing Transformers + base deps"
$PIP_BIN install \
  transformers==4.59.1 accelerate==0.34.2 \
  sentencepiece==0.2.0 protobuf==5.28.2

log "3) Installing ms-swift from PyPI (deps enabled)"
$PIP_BIN install -U ms-swift

log "4) Installing Qwen/PEFT/liger-kernel (deps enabled)"
$PIP_BIN install -U "qwen_vl_utils>=0.0.14" peft liger-kernel

if [[ "$INSTALL_GCC" == "1" ]]; then
  if command -v conda >/dev/null 2>&1; then
    log "5) Installing GCC/GXX 11 via conda"
    conda install -c conda-forge gcc=11 gxx=11 -y
  else
    log "5) Skip GCC install (conda not found)"
  fi
fi

log "6) Installing optional local wheels if present"
if [[ -f "$CAUSAL_CONV_WHL" ]]; then
  $PIP_BIN install "$CAUSAL_CONV_WHL" --no-build-isolation
else
  log "Skip $CAUSAL_CONV_WHL (not found)"
fi

if [[ -f "$FLASH_ATTN_WHL" ]]; then
  $PIP_BIN install "$FLASH_ATTN_WHL" --no-build-isolation
else
  log "Skip $FLASH_ATTN_WHL (not found)"
fi

log "7) Installing deepspeed (deps enabled)"
$PIP_BIN install -U deepspeed

log "8) Installing vLLM"
$PIP_BIN install -U "vllm>=0.17.0"

log "9) Installing local editable ms-swift from repo"
if [[ ! -d "$SWIFT_REPO" ]]; then
  echo "ERROR: SWIFT_REPO does not exist: $SWIFT_REPO" >&2
  exit 1
fi
$PIP_BIN install -e "$SWIFT_REPO" --no-deps

log "10) Smoke test"
$PYTHON_BIN - <<'PY'
import importlib
mods = ["torch", "transformers", "accelerate", "swift", "vllm", "deepspeed"]
for m in mods:
    importlib.import_module(m)
print("Smoke test OK: imported", ", ".join(mods))
PY

log "Done"
