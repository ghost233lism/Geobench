#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEOVERSE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${GEOVERSE_DIR}/.." && pwd)"

CHECKPOINT_PATH="/mnt/data/zhangyiming/database/ckpt/geobench/geogrpo_Qwen3.5-9B_b24_rgeoscore_accuracy_formatstrict_notags1_think0_20260603-162205/v0-20260603-162501/checkpoint-5000"

export MODEL_PATH="${MODEL_PATH:-${CHECKPOINT_PATH}}"
export PROMPT_FILE="${PROMPT_FILE:-${REPO_ROOT}/tools/system_prompt_compact_no_think.txt}"
export USER_PROMPT="${USER_PROMPT:-<image> Based on the image, reason carefully about the visual clues and tell me the specific location. Output only the required JSON.}"

export GPU="${GPU:-0}"
export INFER_BACKEND="${INFER_BACKEND:-vllm}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export LIMIT="${LIMIT:--1}"

export DATA_ROOT="${DATA_ROOT:-${GEOVERSE_DIR}/geobench-val}"
export OUTPUT_DIR="${OUTPUT_DIR:-${GEOVERSE_DIR}/eval_runs/checkpoint5000_1gpu_all_$(date +%Y%m%d_%H%M%S)}"
export SWIFT_VLLM_FORCE_NATIVE_GDN="${SWIFT_VLLM_FORCE_NATIVE_GDN:-1}"

# This checkpoint is full-parameter, not a LoRA adapter. Keep all-data evaluation by default.
export ADAPTERS=""
export INPUT_JSONS=""

exec bash "${GEOVERSE_DIR}/run_eval_geoverse_val_geoscore.sh"
