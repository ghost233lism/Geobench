#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/nas/zhangyiming/geobench}"
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
RUN_SCRIPT="${RUN_SCRIPT:-${REPO_ROOT}/tools/run_geogrpo_A100_cosmos.sh}"

RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
export RUN_NAME="${RUN_NAME:-geogrpo_qwen35_9b_colocate_smoke_1gpu_${RUN_STAMP}}"
export LOG_DIR="${LOG_DIR:-${REPO_ROOT}/output/${RUN_NAME}/logs}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/scheduled/${RUN_NAME}}"
export CHECKPOINT_OUTPUT_ROOT="${CHECKPOINT_OUTPUT_ROOT:-/mnt/data/zhangyiming/database/ckpt/geobench}"
export SWIFT_OUTPUT_DIR="${SWIFT_OUTPUT_DIR:-${CHECKPOINT_OUTPUT_ROOT}/${RUN_NAME}}"
export OFFLOAD_CHECKPOINTS="${OFFLOAD_CHECKPOINTS:-false}"
export INPUT_JSONL="${INPUT_JSONL:-${REPO_ROOT}/all_selected_merged_current_paths.jsonl}"
export MODEL="${MODEL:-${REPO_ROOT}/models/Qwen3.5-9B}"

export CONDA_ENV_NAME="${CONDA_ENV_NAME:-${CONDA_DEFAULT_ENV:-}}"
export CONDA_CLONE_SOURCE="${CONDA_CLONE_SOURCE:-cosmos}"
export REQUIRE_QWEN35_RUNTIME="${REQUIRE_QWEN35_RUNTIME:-false}"
export SKIP_RUNTIME_ENV_CHECK="${SKIP_RUNTIME_ENV_CHECK:-true}"
export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"
export MAX_JOBS="${MAX_JOBS:-8}"

export TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0}"
export TRAIN_NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE:-1}"
export VLLM_MODE="${VLLM_MODE:-colocate}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.20}"
export MOVE_MODEL_BATCHES="${MOVE_MODEL_BATCHES:-1}"
export SLEEP_LEVEL="${SLEEP_LEVEL:-0}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-true}"
export ROLLOUT_TEMPLATE="${ROLLOUT_TEMPLATE:-qwen3_5}"

export MAX_SAMPLES="${MAX_SAMPLES:-4}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export STEPS_PER_GENERATION="${STEPS_PER_GENERATION:-2}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
export DEEPSPEED="${DEEPSPEED:-${REPO_ROOT}/swift/config/zero2_offload.json}"
export MAX_LENGTH="${MAX_LENGTH:-4096}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}"
export ROLLOUT_VLLM_MAX_MODEL_LEN="${ROLLOUT_VLLM_MAX_MODEL_LEN:-4096}"
export MAX_PIXELS="${MAX_PIXELS:-131072}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-2}"
export DATASET_NUM_PROC="${DATASET_NUM_PROC:-1}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"
export TEMPERATURE="${TEMPERATURE:-0.7}"
export TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
export GEOBENCH_STABILIZE_PARAM_GRAD_HOOKS="${GEOBENCH_STABILIZE_PARAM_GRAD_HOOKS:-1}"
export QWEN_ENABLE_THINKING="${QWEN_ENABLE_THINKING:-false}"
export SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-1}"

export REWARD_FUNCS_TEXT="${REWARD_FUNCS_TEXT:-geoscore_accuracy geo_strict_format geo_no_unknown}"
export REWARD_WEIGHTS_TEXT="${REWARD_WEIGHTS_TEXT:-2.0 1.0 1.0}"
export SCHEDULE="${SCHEDULE:-random}"
export DOMAIN_BALANCE="${DOMAIN_BALANCE:-random}"
export DOMAIN_RATIO_TEXT="${DOMAIN_RATIO_TEXT:-}"

DEFAULT_ROLLOUT_VLLM_LIMIT_MM_PER_PROMPT='{"image":1}'
export ROLLOUT_VLLM_LIMIT_MM_PER_PROMPT="${ROLLOUT_VLLM_LIMIT_MM_PER_PROMPT:-${DEFAULT_ROLLOUT_VLLM_LIMIT_MM_PER_PROMPT}}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/tools/tracking_env.sh"
geobench_configure_tracking "${RUN_NAME}"

DEFAULT_EXTRA_SWIFT_ARGS="--report_to ${REPORT_TO} --template qwen3_5 --attn_impl flash_attention_2 --steps_per_generation ${STEPS_PER_GENERATION} --vllm_max_model_len ${ROLLOUT_VLLM_MAX_MODEL_LEN} --vllm_max_num_seqs ${VLLM_MAX_NUM_SEQS} --vllm_enforce_eager ${VLLM_ENFORCE_EAGER} --vllm_limit_mm_per_prompt ${ROLLOUT_VLLM_LIMIT_MM_PER_PROMPT} --fp16 false --bf16 true --optim adamw_torch"
if [[ -n "${EXTRA_SWIFT_ARGS_APPEND_TEXT:-}" ]]; then
  DEFAULT_EXTRA_SWIFT_ARGS="${DEFAULT_EXTRA_SWIFT_ARGS} ${EXTRA_SWIFT_ARGS_APPEND_TEXT}"
fi
export EXTRA_SWIFT_ARGS_TEXT="${EXTRA_SWIFT_ARGS_TEXT:-${DEFAULT_EXTRA_SWIFT_ARGS}}"

mkdir -p "${LOG_DIR}"

echo "Starting GeoBench Qwen3.5-9B 1-GPU colocate smoke run."
echo "Run name: ${RUN_NAME}"
echo "Model: ${MODEL}"
echo "Training GPU: ${TRAIN_CUDA_VISIBLE_DEVICES}; nproc=${TRAIN_NPROC_PER_NODE}; batch=${PER_DEVICE_TRAIN_BATCH_SIZE}; generations=${NUM_GENERATIONS}; max_samples=${MAX_SAMPLES}"
echo "FlashAttention check: attn_impl=flash_attention_2; torch_dtype=${TORCH_DTYPE}; bf16=true"
echo "DeepSpeed: ${DEEPSPEED}"
echo "Colocate: tp=${VLLM_TENSOR_PARALLEL_SIZE}; gpu_memory=${VLLM_GPU_MEMORY_UTILIZATION}; max_model_len=${ROLLOUT_VLLM_MAX_MODEL_LEN}; max_num_seqs=${VLLM_MAX_NUM_SEQS}; enforce_eager=${VLLM_ENFORCE_EAGER}"
echo "Runtime env check skipped: ${SKIP_RUNTIME_ENV_CHECK}; require_qwen35_runtime=${REQUIRE_QWEN35_RUNTIME}; ds_skip_cuda_check=${DS_SKIP_CUDA_CHECK}"
echo "Logs: ${LOG_DIR}"

bash "${RUN_SCRIPT}"
