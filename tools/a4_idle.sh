#!/usr/bin/env bash
set -euo pipefail

# Default background job for the repaired A4 controller. This intentionally
# keeps the 8-card machine busy when no training override is active.
exec env \
  PHYSICAL_GPU_IDS="${PHYSICAL_GPU_IDS:-0,1,2,3,4,5,6,7}" \
  WANDB_MODE="${WANDB_MODE:-offline}" \
  bash /mnt/nas/zhangyiming/train/bash/cosmos_cot8.sh
