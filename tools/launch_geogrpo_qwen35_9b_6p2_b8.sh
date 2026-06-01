#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/nas/zhangyiming/geobench}"
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
RUN_SCRIPT="${RUN_SCRIPT:-${REPO_ROOT}/tools/run_geogrpo_A100_cosmos.sh}"

RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
export RUN_NAME="${RUN_NAME:-geogrpo_qwen35_9b_6p2_b8_${RUN_STAMP}}"
export LOG_DIR="${LOG_DIR:-${REPO_ROOT}/output/${RUN_NAME}/logs}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/scheduled/${RUN_NAME}}"
export CHECKPOINT_OUTPUT_ROOT="${CHECKPOINT_OUTPUT_ROOT:-/mnt/data/zhangyiming/database/ckpt/geobench}"
export SWIFT_OUTPUT_DIR="${SWIFT_OUTPUT_DIR:-${CHECKPOINT_OUTPUT_ROOT}/${RUN_NAME}}"
export OFFLOAD_CHECKPOINTS="${OFFLOAD_CHECKPOINTS:-false}"
export INPUT_JSONL="${INPUT_JSONL:-${REPO_ROOT}/all_selected_merged_current_paths.jsonl}"
export MODEL="${MODEL:-${REPO_ROOT}/models/Qwen3.5-9B}"

export CONDA_ENV_NAME="${CONDA_ENV_NAME:-qwen35b}"
export CONDA_CLONE_SOURCE="${CONDA_CLONE_SOURCE:-cosmos}"
export REQUIRE_QWEN35_RUNTIME="${REQUIRE_QWEN35_RUNTIME:-true}"

export TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export TRAIN_NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE:-6}"
export VLLM_MODE="${VLLM_MODE:-server}"
export ROLLOUT_CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES:-6,7}"
export ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-1}"
export ROLLOUT_DATA_PARALLEL_SIZE="${ROLLOUT_DATA_PARALLEL_SIZE:-2}"
export ROLLOUT_TEMPLATE="${ROLLOUT_TEMPLATE:-qwen3_5}"
export ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION="${ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
export ROLLOUT_VLLM_MAX_MODEL_LEN="${ROLLOUT_VLLM_MAX_MODEL_LEN:-10240}"
export ROLLOUT_VLLM_MAX_NUM_SEQS="${ROLLOUT_VLLM_MAX_NUM_SEQS:-64}"
export ROLLOUT_VLLM_ENFORCE_EAGER="${ROLLOUT_VLLM_ENFORCE_EAGER:-false}"

export NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export STEPS_PER_GENERATION="${STEPS_PER_GENERATION:-1}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
export DEEPSPEED="${DEEPSPEED:-zero3}"
export MAX_LENGTH="${MAX_LENGTH:-10240}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-4096}"
export MAX_PIXELS="${MAX_PIXELS:-262144}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-16384}"
export DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
export TEMPERATURE="${TEMPERATURE:-0.9}"
export QWEN_ENABLE_THINKING="${QWEN_ENABLE_THINKING:-false}"
export SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

export REWARD_FUNCS_TEXT="${REWARD_FUNCS_TEXT:-format geoscore_accuracy geo_format}"
export REWARD_WEIGHTS_TEXT="${REWARD_WEIGHTS_TEXT:-1.0 1.0 1.0}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/tools/tracking_env.sh"
geobench_configure_tracking "${RUN_NAME}"

DEFAULT_EXTRA_SWIFT_ARGS="--report_to ${REPORT_TO} --template qwen3_5 --attn_impl sdpa --steps_per_generation ${STEPS_PER_GENERATION} --save_strategy ${SAVE_STRATEGY} --save_total_limit ${SAVE_TOTAL_LIMIT}"
export EXTRA_SWIFT_ARGS_TEXT="${EXTRA_SWIFT_ARGS_TEXT:-${DEFAULT_EXTRA_SWIFT_ARGS}}"

mkdir -p "${LOG_DIR}"
GPU_MONITOR_LOG="${GPU_MONITOR_LOG:-${LOG_DIR}/nvidia-smi.csv}"
monitor_pid=""
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu,power.draw \
    --format=csv -l 5 >"${GPU_MONITOR_LOG}" 2>&1 &
  monitor_pid="$!"
fi
cleanup_monitor() {
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" >/dev/null 2>&1 || true
    wait "${monitor_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup_monitor EXIT

W="${TRAIN_NPROC_PER_NODE}"
B="${PER_DEVICE_TRAIN_BATCH_SIZE}"
G="${NUM_GENERATIONS}"
S="${STEPS_PER_GENERATION}"
generation_completions=$((W * B * S))
step_completions=$((W * B))
generation_prompts=$((generation_completions / G))
step_prompts=$((step_completions / G))

echo "Starting GeoBench Qwen3.5-9B 6+2 run."
echo "Run name: ${RUN_NAME}"
echo "Model: ${MODEL}"
echo "Training GPUs: ${TRAIN_CUDA_VISIBLE_DEVICES}; W=${W}; per_device_train_batch_size=${B}; grad_accum=${GRADIENT_ACCUMULATION_STEPS}"
echo "Rollout GPUs: ${ROLLOUT_CUDA_VISIBLE_DEVICES}; TP=${ROLLOUT_TENSOR_PARALLEL_SIZE}; DP=${ROLLOUT_DATA_PARALLEL_SIZE}; max_num_seqs=${ROLLOUT_VLLM_MAX_NUM_SEQS}; enforce_eager=${ROLLOUT_VLLM_ENFORCE_EAGER}"
echo "G=${G}; S=${S}; generation batch=${generation_completions} completions=${generation_prompts} prompts; step batch=${step_completions} completions=${step_prompts} prompts"
echo "Input: ${INPUT_JSONL}; max_samples=${MAX_SAMPLES:-all}"
echo "Swift output/checkpoint dir: ${SWIFT_OUTPUT_DIR}; offload=${OFFLOAD_CHECKPOINTS}"
echo "GPU monitor: ${GPU_MONITOR_LOG}"

exec bash "${RUN_SCRIPT}"
