#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/path/to/geobench}"
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
export REPO_ROOT
RUN_SCRIPT="${RUN_SCRIPT:-${REPO_ROOT}/tools/run_geogrpo_A100_cosmos.sh}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
USER_RUN_NAME="${RUN_NAME:-}"
export CHECKPOINT_OUTPUT_ROOT="${CHECKPOINT_OUTPUT_ROOT:-/path/to/checkpoints}"
export OFFLOAD_CHECKPOINTS="${OFFLOAD_CHECKPOINTS:-false}"
export INPUT_JSONL="${INPUT_JSONL:-/path/to/train_data.jsonl}"
export TRAIN_DATA_RATIO="${TRAIN_DATA_RATIO:-1.0}"
export MODEL="${MODEL:-/path/to/Qwen3.5-9B}"
export QWEN35_NO_TAGS="${QWEN35_NO_TAGS:-true}"
case "${QWEN35_NO_TAGS,,}" in
  1|true|yes|on) qwen35_no_tags_enabled=true ;;
  *) qwen35_no_tags_enabled=false ;;
esac

export CONDA_ENV_NAME="${CONDA_ENV_NAME:-${CONDA_DEFAULT_ENV:-}}"
export CONDA_CLONE_SOURCE="${CONDA_CLONE_SOURCE:-cosmos}"
export REQUIRE_QWEN35_RUNTIME="${REQUIRE_QWEN35_RUNTIME:-false}"
export SKIP_RUNTIME_ENV_CHECK="${SKIP_RUNTIME_ENV_CHECK:-true}"

export TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export TRAIN_NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE:-6}"
export VLLM_MODE="${VLLM_MODE:-server}"
export ROLLOUT_CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES:-6,7}"
export ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-2}"
export ROLLOUT_DATA_PARALLEL_SIZE="${ROLLOUT_DATA_PARALLEL_SIZE:-1}"
export ROLLOUT_TEMPLATE="${ROLLOUT_TEMPLATE:-qwen3_5}"
export ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION="${ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
export ROLLOUT_VLLM_MAX_MODEL_LEN="${ROLLOUT_VLLM_MAX_MODEL_LEN:-8192}"
# Qwen3.5 uses mixed full attention + GDN/linear attention. In this runtime stack
# the non-eager vLLM V1 path is brittle under server weight sync, so keep the
# server rollout on the same conservative settings as the stable colocate run.
export ROLLOUT_VLLM_MAX_NUM_SEQS="${ROLLOUT_VLLM_MAX_NUM_SEQS:-192}"
export ROLLOUT_VLLM_ENFORCE_EAGER="${ROLLOUT_VLLM_ENFORCE_EAGER:-false}"

export NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-24}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export STEPS_PER_GENERATION="${STEPS_PER_GENERATION:-1}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
export DEEPSPEED="${DEEPSPEED:-${REPO_ROOT}/swift/config/zero3_bf16.json}"
export MAX_LENGTH="${MAX_LENGTH:-8192}"
if [[ "${qwen35_no_tags_enabled}" == "true" ]]; then
  export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
  export SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-${REPO_ROOT}/tools/system_prompt_compact_no_think.txt}"
  export QWEN35_USER_PROMPT="${QWEN35_USER_PROMPT:-<image> Based on the image, reason carefully about the visual clues and tell me the specific location. Output only the required JSON.}"
else
  export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
fi
export MAX_PIXELS="${MAX_PIXELS:-802816}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-8192}"
export DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
export TEMPERATURE="${TEMPERATURE:-0.9}"
export TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
export GEOBENCH_LOG_SITECUSTOMIZE="${GEOBENCH_LOG_SITECUSTOMIZE:-1}"
export GEOBENCH_PATCH_QWEN35_ZERO3_CONV1D="${GEOBENCH_PATCH_QWEN35_ZERO3_CONV1D:-1}"
export GEOBENCH_DISABLE_QWEN35_CAUSAL_CONV1D="${GEOBENCH_DISABLE_QWEN35_CAUSAL_CONV1D:-1}"
# On A4/Hopper, Qwen3.5 bf16 ZeRO-3 can produce non-finite grads on the first
# optimizer step. Stabilize them and fail before vLLM sync if weights are bad.
export GEOBENCH_STABILIZE_PARAM_GRAD_HOOKS="${GEOBENCH_STABILIZE_PARAM_GRAD_HOOKS:-1}"
export GEOBENCH_LOG_PARAM_GRAD_HOOKS="${GEOBENCH_LOG_PARAM_GRAD_HOOKS:-1}"
export GEOBENCH_LOG_PARAM_GRAD_HOOK_MAX_LOGS="${GEOBENCH_LOG_PARAM_GRAD_HOOK_MAX_LOGS:-40}"
export GEOBENCH_LOG_VLLM_SYNC_FINITE="${GEOBENCH_LOG_VLLM_SYNC_FINITE:-0}"
export GEOBENCH_ABORT_ON_NONFINITE_VLLM_SYNC="${GEOBENCH_ABORT_ON_NONFINITE_VLLM_SYNC:-0}"
export QWEN_ENABLE_THINKING="${QWEN_ENABLE_THINKING:-false}"
export SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

if [[ "${qwen35_no_tags_enabled}" == "true" ]]; then
  export REWARD_FUNCS_TEXT="${REWARD_FUNCS_TEXT:-geoscore_accuracy formatstrict}"
  export REWARD_WEIGHTS_TEXT="${REWARD_WEIGHTS_TEXT:-2.0 1.0}"
else
  export REWARD_FUNCS_TEXT="${REWARD_FUNCS_TEXT:-geoscore_accuracy geo_strict_format}"
  export REWARD_WEIGHTS_TEXT="${REWARD_WEIGHTS_TEXT:-2.0 1.0}"
fi

sanitize_run_name_part() {
  local value="$1"
  value="${value// /_}"
  value="$(printf '%s' "${value}" | tr -cs '[:alnum:]_.=-' '_' | sed -E 's/^_+|_+$//g; s/_+/_/g')"
  printf '%s' "${value:-none}"
}

if [[ -n "${USER_RUN_NAME}" ]]; then
  export RUN_NAME="${USER_RUN_NAME}"
else
  model_part="$(sanitize_run_name_part "$(basename "${MODEL}")")"
  rewards_part="$(sanitize_run_name_part "${REWARD_FUNCS_TEXT}")"
  no_tags_part="notags0"
  if [[ "${qwen35_no_tags_enabled}" == "true" ]]; then
    no_tags_part="notags1"
  fi
  thinking_part="think0"
  case "${QWEN_ENABLE_THINKING,,}" in
    1|true|yes|on) thinking_part="think1" ;;
  esac
  export RUN_NAME="geogrpo_${model_part}_b${PER_DEVICE_TRAIN_BATCH_SIZE}_r${rewards_part}_${no_tags_part}_${thinking_part}_${RUN_STAMP}"
fi

export LOG_DIR="${LOG_DIR:-${REPO_ROOT}/output/${RUN_NAME}/logs}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/scheduled/${RUN_NAME}}"
export SWIFT_OUTPUT_DIR="${SWIFT_OUTPUT_DIR:-${CHECKPOINT_OUTPUT_ROOT}/${RUN_NAME}}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/tools/tracking_env.sh"
geobench_configure_tracking "${RUN_NAME}"

DEFAULT_EXTRA_SWIFT_ARGS="--report_to ${REPORT_TO} --template qwen3_5 --attn_impl flash_attention_2 --steps_per_generation ${STEPS_PER_GENERATION} --save_strategy ${SAVE_STRATEGY} --save_total_limit ${SAVE_TOTAL_LIMIT} --fp16 false --bf16 true --optim adamw_torch"
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

echo "Starting GeoBench Qwen3.5-9B 6+2 run."
echo "Run name: ${RUN_NAME}"
echo "Model: ${MODEL}"
echo "Qwen3.5 no-tags JSON mode: ${QWEN35_NO_TAGS}; enable_thinking=${QWEN_ENABLE_THINKING}"
echo "System prompt file: ${SYSTEM_PROMPT_FILE:-<default>}"
echo "User prompt: ${QWEN35_USER_PROMPT:-<default>}"
echo "Training GPUs: ${TRAIN_CUDA_VISIBLE_DEVICES}; W=${W}; per_device_train_batch_size=${B}; grad_accum=${GRADIENT_ACCUMULATION_STEPS}"
echo "Torch dtype: ${TORCH_DTYPE}"
echo "DeepSpeed: ${DEEPSPEED}"
echo "Qwen3.5 ZeRO-3 conv1d patch: ${GEOBENCH_PATCH_QWEN35_ZERO3_CONV1D}; disable causal_conv1d=${GEOBENCH_DISABLE_QWEN35_CAUSAL_CONV1D}"
echo "GeoBench sitecustomize logging: ${GEOBENCH_LOG_SITECUSTOMIZE}"
echo "Grad finite scan hook: stabilize=${GEOBENCH_STABILIZE_PARAM_GRAD_HOOKS}; log=${GEOBENCH_LOG_PARAM_GRAD_HOOKS}; vLLM_sync_finite=${GEOBENCH_LOG_VLLM_SYNC_FINITE}; abort_sync=${GEOBENCH_ABORT_ON_NONFINITE_VLLM_SYNC}"
echo "Rollout GPUs: ${ROLLOUT_CUDA_VISIBLE_DEVICES}; TP=${ROLLOUT_TENSOR_PARALLEL_SIZE}; DP=${ROLLOUT_DATA_PARALLEL_SIZE}; max_num_seqs=${ROLLOUT_VLLM_MAX_NUM_SEQS}; enforce_eager=${ROLLOUT_VLLM_ENFORCE_EAGER}"
echo "G=${G}; S=${S}; generation batch=${generation_completions} completions=${generation_prompts} prompts; step batch=${step_completions} completions=${step_prompts} prompts"
echo "Input: ${INPUT_JSONL}; train_data_ratio=${TRAIN_DATA_RATIO}; max_samples=${MAX_SAMPLES:-all}"
echo "Swift output/checkpoint dir: ${SWIFT_OUTPUT_DIR}; offload=${OFFLOAD_CHECKPOINTS}"
echo "GPU monitor: ${GPU_MONITOR_LOG}"

exec bash "${RUN_SCRIPT}"
