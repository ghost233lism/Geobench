#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/nas/zhangyiming/geobench}"
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
RUN_SCRIPT="${RUN_SCRIPT:-${REPO_ROOT}/tools/run_geogrpo_A100_cosmos.sh}"

RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
export RUN_NAME="${RUN_NAME:-geogrpo_qwen35_9b_colocate_b12_${RUN_STAMP}}"
export LOG_DIR="${LOG_DIR:-${REPO_ROOT}/output/${RUN_NAME}/logs}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/scheduled/${RUN_NAME}}"
export CHECKPOINT_OUTPUT_ROOT="${CHECKPOINT_OUTPUT_ROOT:-/mnt/data/zhangyiming/database/ckpt/geobench}"
export SWIFT_OUTPUT_DIR="${SWIFT_OUTPUT_DIR:-${CHECKPOINT_OUTPUT_ROOT}/${RUN_NAME}}"
export OFFLOAD_CHECKPOINTS="${OFFLOAD_CHECKPOINTS:-false}"
export INPUT_JSONL="${INPUT_JSONL:-${REPO_ROOT}/all_selected_merged_current_paths.jsonl}"
export MODEL="${MODEL:-${REPO_ROOT}/models/Qwen3.5-9B}"

export CONDA_ENV_NAME="${CONDA_ENV_NAME:-qwen35b}"
export CONDA_CLONE_SOURCE="${CONDA_CLONE_SOURCE:-cosmos}"
export REQUIRE_QWEN35_RUNTIME="${REQUIRE_QWEN35_RUNTIME:-false}"
export SKIP_RUNTIME_ENV_CHECK="${SKIP_RUNTIME_ENV_CHECK:-true}"

export TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TRAIN_NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE:-8}"
export VLLM_MODE="${VLLM_MODE:-colocate}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
# Verified on A4 8x GPU with per-device batch 12. Raising this together with
# vLLM max seqs pushes the run much closer to the memory edge.
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.30}"
export MOVE_MODEL_BATCHES="${MOVE_MODEL_BATCHES:-8}"
export ROLLOUT_TEMPLATE="${ROLLOUT_TEMPLATE:-qwen3_5}"

export NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-12}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export STEPS_PER_GENERATION="${STEPS_PER_GENERATION:-1}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
export DEEPSPEED="${DEEPSPEED:-${REPO_ROOT}/swift/config/zero3_bf16.json}"
export MAX_LENGTH="${MAX_LENGTH:-10240}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"
export MAX_PIXELS="${MAX_PIXELS:-262144}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-8192}"
export DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
export TEMPERATURE="${TEMPERATURE:-0.9}"
# On A4/Hopper, Qwen3.5 bf16 ZeRO-3 needs the grad finite scan hook below;
# without it the first optimizer step can corrupt language weights to NaN.
export TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
export GEOBENCH_STABILIZE_PARAM_GRAD_HOOKS="${GEOBENCH_STABILIZE_PARAM_GRAD_HOOKS:-1}"
export QWEN_ENABLE_THINKING="${QWEN_ENABLE_THINKING:-false}"
export SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
export SAVE_STEPS="${SAVE_STEPS:-5000}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.01}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-1}"

export REWARD_FUNCS_TEXT="${REWARD_FUNCS_TEXT:-geoscore_accuracy geo_strict_format geo_no_unknown}"
export REWARD_WEIGHTS_TEXT="${REWARD_WEIGHTS_TEXT:-2.0 1.0 1.0}"
export SCHEDULE="${SCHEDULE:-four_stage}"
export DOMAIN_BALANCE="${DOMAIN_BALANCE:-ratio}"
export DOMAIN_RATIO_TEXT="${DOMAIN_RATIO_TEXT:-}"

export ROLLOUT_VLLM_MAX_MODEL_LEN="${ROLLOUT_VLLM_MAX_MODEL_LEN:-10240}"
DEFAULT_ROLLOUT_VLLM_LIMIT_MM_PER_PROMPT='{"image":1}'
export ROLLOUT_VLLM_LIMIT_MM_PER_PROMPT="${ROLLOUT_VLLM_LIMIT_MM_PER_PROMPT:-${DEFAULT_ROLLOUT_VLLM_LIMIT_MM_PER_PROMPT}}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
export SLEEP_LEVEL="${SLEEP_LEVEL:-0}"
# Qwen3.5 colocate currently fails during vLLM V1 sampler/KV-cache profiling
# on the non-eager path in this runtime stack.
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-true}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/tools/tracking_env.sh"
geobench_configure_tracking "${RUN_NAME}"

DEFAULT_EXTRA_SWIFT_ARGS="--report_to ${REPORT_TO} --template qwen3_5 --attn_impl flash_attention_2 --steps_per_generation ${STEPS_PER_GENERATION} --vllm_max_model_len ${ROLLOUT_VLLM_MAX_MODEL_LEN} --vllm_max_num_seqs ${VLLM_MAX_NUM_SEQS} --vllm_enforce_eager ${VLLM_ENFORCE_EAGER} --vllm_limit_mm_per_prompt ${ROLLOUT_VLLM_LIMIT_MM_PER_PROMPT} --fp16 false --bf16 true --optim adamw_torch"
if [[ -n "${EXTRA_SWIFT_ARGS_APPEND_TEXT:-}" ]]; then
  DEFAULT_EXTRA_SWIFT_ARGS="${DEFAULT_EXTRA_SWIFT_ARGS} ${EXTRA_SWIFT_ARGS_APPEND_TEXT}"
fi
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

echo "Starting GeoBench Qwen3.5-9B colocate run."
echo "Run name: ${RUN_NAME}"
echo "Model: ${MODEL}"
echo "Training GPUs: ${TRAIN_CUDA_VISIBLE_DEVICES}; W=${W}; per_device_train_batch_size=${B}; grad_accum=${GRADIENT_ACCUMULATION_STEPS}"
echo "Torch dtype: ${TORCH_DTYPE}"
echo "DeepSpeed: ${DEEPSPEED}"
echo "Learning rate: ${LEARNING_RATE}; warmup_ratio=${WARMUP_RATIO}; max_grad_norm=${MAX_GRAD_NORM}"
echo "Grad finite scan hook: ${GEOBENCH_STABILIZE_PARAM_GRAD_HOOKS}"
echo "Colocate: tp=${VLLM_TENSOR_PARALLEL_SIZE}; gpu_memory=${VLLM_GPU_MEMORY_UTILIZATION}; max_num_seqs=${VLLM_MAX_NUM_SEQS}; enforce_eager=${VLLM_ENFORCE_EAGER}; sleep_level=${SLEEP_LEVEL}"
echo "G=${G}; S=${S}; generation batch=${generation_completions} completions=${generation_prompts} prompts; step batch=${step_completions} completions=${step_prompts} prompts"
echo "Input: ${INPUT_JSONL}; max_samples=${MAX_SAMPLES:-all}"
echo "Schedule: ${SCHEDULE}; domain_balance=${DOMAIN_BALANCE}; domain_ratio=${DOMAIN_RATIO_TEXT:-default}"
echo "Swift output/checkpoint dir: ${SWIFT_OUTPUT_DIR}; offload=${OFFLOAD_CHECKPOINTS}"
echo "GPU monitor: ${GPU_MONITOR_LOG}"

bash "${RUN_SCRIPT}"
