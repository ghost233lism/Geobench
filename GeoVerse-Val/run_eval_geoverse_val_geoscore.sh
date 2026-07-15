#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="${MODE:-run}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/geobench-val/all}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/eval_runs/$(date +%Y%m%d_%H%M%S)}"
PROMPT_FILE="${PROMPT_FILE:-${REPO_ROOT}/tools/system_prompt.txt}"
USER_PROMPT="${USER_PROMPT:-<image> Based on the image, tell me the specific location and your thinking process. Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.}"
GEOSCORE_API_KEY_CONFIG="${GEOSCORE_API_KEY_CONFIG:-${REPO_ROOT}/tools/geoscore_api_keys.conf}"
GEOSCORE_CACHE_FILE="${GEOSCORE_CACHE_FILE:-${OUTPUT_DIR}/geoscore_cache.json}"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/zhangyiming/database/ckpt/geobench/geogrpo_Qwen3.5-9B_b24_rgeoscore_accuracy_formatstrict_notags1_think0_schedule_four_stage_domain_balance_random_resume2000_20260620-001026/v3-20260627-110357/checkpoint-5000}"
ADAPTERS="${ADAPTERS:-}"
INPUT_JSONS="${INPUT_JSONS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
INFER_BACKEND="${INFER_BACKEND:-vllm}"
GEOBENCH_FORCE_NATIVE_GDN="${GEOBENCH_FORCE_NATIVE_GDN:-1}"
VLLM_ENGINE_KWARGS="${VLLM_ENGINE_KWARGS:-{\"enable_flashinfer_autotune\":false}}"

BATCH_SIZE="${BATCH_SIZE:-64}"
BATCH_TIMEOUT_SEC="${BATCH_TIMEOUT_SEC:-3000}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1}"
TOP_K="${TOP_K:--1}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-}"
MODEL_TYPE="${MODEL_TYPE:-qwen3_5}"
TEMPLATE_TYPE="${TEMPLATE_TYPE:-qwen3_5}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\": 1}}"
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"
LIMIT="${LIMIT:--1}"
IMAGE_FIELD="${IMAGE_FIELD:-image_path}"

ANSWER_FIELD="${ANSWER_FIELD:-}"
SUCCESS_FIELD="${SUCCESS_FIELD:-}"
RESULT_FIELD="${RESULT_FIELD:-}"
RESULT_SUFFIX="${RESULT_SUFFIX:-_geo}"
GEOSCORE_MAX_DISTANCE="${GEOSCORE_MAX_DISTANCE:-18050.0}"
GEOCODE_TIMEOUT="${GEOCODE_TIMEOUT:-10}"

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS=(
  "--data-root" "${DATA_ROOT}"
  "--output-dir" "${OUTPUT_DIR}"
  "--image-field" "${IMAGE_FIELD}"
  "--limit" "${LIMIT}"
)

if [[ -n "${INPUT_JSONS}" ]]; then
  read -r -a INPUT_JSON_ARR <<< "${INPUT_JSONS}"
  COMMON_ARGS+=("--input-json" "${INPUT_JSON_ARR[@]}")
fi

INFER_ARGS=(
  "--gpu" "${GPU}"
  "--batch-size" "${BATCH_SIZE}"
  "--batch-timeout-sec" "${BATCH_TIMEOUT_SEC}"
  "--max-tokens" "${MAX_TOKENS}"
  "--temperature" "${TEMPERATURE}"
  "--top-p" "${TOP_P}"
  "--top-k" "${TOP_K}"
  "--device-map" "${DEVICE_MAP}"
  "--infer-backend" "${INFER_BACKEND}"
  "--prompt-file" "${PROMPT_FILE}"
  "--user-prompt" "${USER_PROMPT}"
)

if [[ -n "${MODEL_PATH}" ]]; then
  INFER_ARGS+=("--model-path" "${MODEL_PATH}")
fi
if [[ -n "${ADAPTERS}" ]]; then
  read -r -a ADAPTER_ARR <<< "${ADAPTERS}"
  INFER_ARGS+=("--adapters" "${ADAPTER_ARR[@]}")
fi
if [[ -n "${TORCH_DTYPE}" ]]; then
  INFER_ARGS+=("--torch-dtype" "${TORCH_DTYPE}")
fi
if [[ -n "${MODEL_TYPE}" ]]; then
  INFER_ARGS+=("--model-type" "${MODEL_TYPE}")
fi
if [[ -n "${TEMPLATE_TYPE}" ]]; then
  INFER_ARGS+=("--template-type" "${TEMPLATE_TYPE}")
fi
if [[ -n "${MAX_MODEL_LEN}" ]]; then
  INFER_ARGS+=("--max-model-len" "${MAX_MODEL_LEN}")
fi
if [[ -n "${MAX_NUM_SEQS}" ]]; then
  INFER_ARGS+=("--max-num-seqs" "${MAX_NUM_SEQS}")
fi
if [[ -n "${GPU_MEMORY_UTILIZATION}" ]]; then
  INFER_ARGS+=("--gpu-memory-utilization" "${GPU_MEMORY_UTILIZATION}")
fi
if [[ -n "${TENSOR_PARALLEL_SIZE}" ]]; then
  INFER_ARGS+=("--tensor-parallel-size" "${TENSOR_PARALLEL_SIZE}")
fi
if [[ -n "${LIMIT_MM_PER_PROMPT}" ]]; then
  INFER_ARGS+=("--limit-mm-per-prompt" "${LIMIT_MM_PER_PROMPT}")
fi
if [[ -n "${VLLM_ENGINE_KWARGS}" ]]; then
  INFER_ARGS+=("--vllm-engine-kwargs" "${VLLM_ENGINE_KWARGS}")
fi
if [[ "${ENFORCE_EAGER}" == "true" ]]; then
  INFER_ARGS+=("--enforce-eager")
fi
if [[ -n "${ANSWER_FIELD}" ]]; then
  INFER_ARGS+=("--answer-field" "${ANSWER_FIELD}")
fi
if [[ -n "${SUCCESS_FIELD}" ]]; then
  INFER_ARGS+=("--success-field" "${SUCCESS_FIELD}")
fi
if [[ "${DISABLE_MODEL_THINKING:-false}" == "true" ]]; then
  INFER_ARGS+=("--disable-model-thinking")
fi

SCORE_ARGS=(
  "--result-suffix" "${RESULT_SUFFIX}"
  "--geoscore-api-key-config" "${GEOSCORE_API_KEY_CONFIG}"
  "--geoscore-cache-file" "${GEOSCORE_CACHE_FILE}"
  "--geoscore-max-distance" "${GEOSCORE_MAX_DISTANCE}"
  "--geocode-timeout" "${GEOCODE_TIMEOUT}"
)

if [[ -n "${SUCCESS_FIELD}" ]]; then
  SCORE_ARGS+=("--success-field" "${SUCCESS_FIELD}")
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
  read -r -a EXTRA_ARGS_ARR <<< "${EXTRA_ARGS}"
else
  EXTRA_ARGS_ARR=()
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export GEOBENCH_FORCE_NATIVE_GDN

case "${MODE}" in
  run)
  if [[ -z "${MODEL_PATH}" && -z "${ADAPTERS}" ]]; then
    echo "MODE=run requires MODEL_PATH or ADAPTERS." >&2
    echo "Example: MODEL_PATH=/path/to/full_model bash $0" >&2
    exit 2
  fi
    python "${SCRIPT_DIR}/eval_geoverse_val_geoscore.py" run \
      "${COMMON_ARGS[@]}" \
      "${INFER_ARGS[@]}" \
      "${SCORE_ARGS[@]}" \
      "${EXTRA_ARGS_ARR[@]}"
    ;;
  infer)
  if [[ -z "${MODEL_PATH}" && -z "${ADAPTERS}" ]]; then
    echo "MODE=infer requires MODEL_PATH or ADAPTERS." >&2
    exit 2
  fi
    python "${SCRIPT_DIR}/eval_geoverse_val_geoscore.py" infer \
      "${COMMON_ARGS[@]}" \
      "${INFER_ARGS[@]}" \
      "${EXTRA_ARGS_ARR[@]}"
    ;;
  score)
    if [[ -z "${ANSWER_FIELD}" ]]; then
      echo "MODE=score requires ANSWER_FIELD." >&2
      exit 2
    fi
    python "${SCRIPT_DIR}/eval_geoverse_val_geoscore.py" score \
      "${COMMON_ARGS[@]}" \
      "--answer-field" "${ANSWER_FIELD}" \
      "${SCORE_ARGS[@]}" \
      "${EXTRA_ARGS_ARR[@]}"
    ;;
  report)
    if [[ -z "${RESULT_FIELD}" ]]; then
      echo "MODE=report requires RESULT_FIELD." >&2
      exit 2
    fi
    python "${SCRIPT_DIR}/eval_geoverse_val_geoscore.py" report \
      "${COMMON_ARGS[@]}" \
      "--result-field" "${RESULT_FIELD}" \
      "${EXTRA_ARGS_ARR[@]}"
    ;;
  *)
    echo "Unknown MODE=${MODE}. Use run, infer, score, or report." >&2
    exit 2
    ;;
esac

echo "Outputs are in: ${OUTPUT_DIR}"
