#!/usr/bin/env bash
set -euo pipefail

# source /root/sunboyuan/miniconda3/bin/activate qwen3.5

cd /nfs/sunboyuan/Geobench/ms-swift

RUN_NAME="${RUN_NAME:-geogrpo_qwen35_2b_four_stage}"
MODEL="${MODEL:-/nfs/sunboyuan/model/Qwen3.5-2B}"
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-/nfs/sunboyuan/Geobench/ms-swift/tools/system_prompt.txt}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1}"
TRAIN_NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE:-2}"
ROLLOUT_CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES:-2,3}"
ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-2}"
ROLLOUT_DATA_PARALLEL_SIZE="${ROLLOUT_DATA_PARALLEL_SIZE:-1}"
ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION="${ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
ROLLOUT_VLLM_MAX_MODEL_LEN="${ROLLOUT_VLLM_MAX_MODEL_LEN:-10240}"
ROLLOUT_VLLM_MAX_NUM_SEQS="${ROLLOUT_VLLM_MAX_NUM_SEQS:-128}"
# Controls Qwen3.5 native thinking mode only. false still keeps the non-thinking <think></think> prefix.
QWEN_ENABLE_THINKING="${QWEN_ENABLE_THINKING:-false}"
# Empty string disables Swift's automatic Qwen response prefix, so the model must emit <think>...</think> itself.
QWEN_RESPONSE_PREFIX="${QWEN_RESPONSE_PREFIX:-}"
ROLLOUT_HOST="${ROLLOUT_HOST:-127.0.0.1}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8000}"
ROLLOUT_WAIT_SECONDS="${ROLLOUT_WAIT_SECONDS:-600}"
LOG_DIR="${LOG_DIR:-/nfs/sunboyuan/Geobench/ms-swift/output/${RUN_NAME}/logs}"
ROLLOUT_LOG="${ROLLOUT_LOG:-${LOG_DIR}/rollout.log}"
TRAIN_LOG="${TRAIN_LOG:-${LOG_DIR}/train.log}"
GEOSCORE_API_KEY_CONFIG="${GEOSCORE_API_KEY_CONFIG:-/nfs/sunboyuan/Geobench/ms-swift/tools/geoscore_api_keys.conf}"
if [[ -f "${GEOSCORE_API_KEY_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${GEOSCORE_API_KEY_CONFIG}"
fi
if [[ -z "${GEOSCORE_API_KEYS:-}" ]]; then
  echo "GEOSCORE_API_KEYS is not set. Put it in ${GEOSCORE_API_KEY_CONFIG} or export it before running." >&2
  exit 1
fi
GEOSCORE_MAX_DISTANCE="${GEOSCORE_MAX_DISTANCE:-18050.0}"
GEOSCORE_CACHE_FILE="${GEOSCORE_CACHE_FILE:-${LOG_DIR}/geoscore_cache.json}"
read -r -a GEOSCORE_API_KEY_ARGS <<< "${GEOSCORE_API_KEYS//,/ }"
export GEOSCORE_CACHE_FILE

mkdir -p "${LOG_DIR}"
SCRIPT_PGID="$(ps -o pgid= -p "$$" | tr -d ' ')"
CLEANING_UP=0
TRAIN_PID=""
TRAIN_PGID=""
ROLLOUT_MONITOR_PID=""

rollout_ready() {
  local url="http://${ROLLOUT_HOST}:${ROLLOUT_PORT}/health/"
  if command -v curl >/dev/null 2>&1; then
    curl -fsS "${url}" >/dev/null 2>&1
    return
  fi
  python - "${url}" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

urllib.request.urlopen(sys.argv[1], timeout=2).read()
PY
}

sync_rollout_port_from_log() {
  local actual_port
  actual_port="$(
    python - "${ROLLOUT_LOG}" <<'PY' 2>/dev/null || true
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
if not log_path.exists():
    raise SystemExit
text = log_path.read_text(errors='ignore')
matches = re.findall(r'Uvicorn running on http://[^:]+:(\d+)', text)
if matches:
    print(matches[-1])
PY
  )"
  if [[ -n "${actual_port}" && "${actual_port}" != "${ROLLOUT_PORT}" ]]; then
    echo "Rollout server selected port ${actual_port} instead of requested ${ROLLOUT_PORT}; using ${actual_port}."
    ROLLOUT_PORT="${actual_port}"
  fi
}

cleanup_rollout() {
  if [[ -n "${ROLLOUT_PGID:-}" ]] && kill -0 -- "-${ROLLOUT_PGID}" >/dev/null 2>&1; then
    if [[ "${ROLLOUT_PGID}" != "${SCRIPT_PGID}" ]]; then
      echo "Stopping rollout process group pgid=${ROLLOUT_PGID}"
      kill -TERM -- "-${ROLLOUT_PGID}" >/dev/null 2>&1 || true
      for _ in {1..10}; do
        kill -0 -- "-${ROLLOUT_PGID}" >/dev/null 2>&1 || break
        sleep 1
      done
      kill -0 -- "-${ROLLOUT_PGID}" >/dev/null 2>&1 && kill -KILL -- "-${ROLLOUT_PGID}" >/dev/null 2>&1 || true
      if [[ -n "${ROLLOUT_PID:-}" ]]; then
        wait "${ROLLOUT_PID}" >/dev/null 2>&1 || true
      fi
    else
      echo "Refusing to kill current shell process group pgid=${ROLLOUT_PGID}; stopping rollout pid=${ROLLOUT_PID:-unknown}"
      kill -TERM "${ROLLOUT_PID}" >/dev/null 2>&1 || true
      wait "${ROLLOUT_PID}" >/dev/null 2>&1 || true
    fi
  elif [[ -n "${ROLLOUT_PID:-}" ]] && kill -0 "${ROLLOUT_PID}" >/dev/null 2>&1; then
    echo "Stopping rollout server pid=${ROLLOUT_PID}"
    kill -TERM "${ROLLOUT_PID}" >/dev/null 2>&1 || true
    for _ in {1..10}; do
      kill -0 "${ROLLOUT_PID}" >/dev/null 2>&1 || break
      sleep 1
    done
    kill -0 "${ROLLOUT_PID}" >/dev/null 2>&1 && kill -KILL "${ROLLOUT_PID}" >/dev/null 2>&1 || true
    wait "${ROLLOUT_PID}" >/dev/null 2>&1 || true
  fi
}

start_rollout_monitor() {
  if [[ -z "${ROLLOUT_PGID:-}" || "${ROLLOUT_PGID}" == "${SCRIPT_PGID}" ]]; then
    return
  fi
  setsid bash -c '
parent_pid="$1"
rollout_pgid="$2"
while kill -0 "${parent_pid}" >/dev/null 2>&1; do
  sleep 2
done
if kill -0 -- "-${rollout_pgid}" >/dev/null 2>&1; then
  echo "Parent script exited; stopping rollout process group pgid=${rollout_pgid}" >&2
  kill -TERM -- "-${rollout_pgid}" >/dev/null 2>&1 || true
  for _ in {1..10}; do
    kill -0 -- "-${rollout_pgid}" >/dev/null 2>&1 || exit 0
    sleep 1
  done
  kill -KILL -- "-${rollout_pgid}" >/dev/null 2>&1 || true
fi
' bash "$$" "${ROLLOUT_PGID}" &
  ROLLOUT_MONITOR_PID=$!
}

cleanup_training() {
  if [[ -n "${TRAIN_PGID:-}" ]] && kill -0 -- "-${TRAIN_PGID}" >/dev/null 2>&1; then
    if [[ "${TRAIN_PGID}" != "${SCRIPT_PGID}" ]]; then
      echo "Stopping training process group pgid=${TRAIN_PGID}"
      kill -TERM -- "-${TRAIN_PGID}" >/dev/null 2>&1 || true
      for _ in {1..10}; do
        kill -0 -- "-${TRAIN_PGID}" >/dev/null 2>&1 || break
        sleep 1
      done
      kill -0 -- "-${TRAIN_PGID}" >/dev/null 2>&1 && kill -KILL -- "-${TRAIN_PGID}" >/dev/null 2>&1 || true
      if [[ -n "${TRAIN_PID:-}" ]]; then
        wait "${TRAIN_PID}" >/dev/null 2>&1 || true
      fi
    fi
  elif [[ -n "${TRAIN_PID:-}" ]] && kill -0 "${TRAIN_PID}" >/dev/null 2>&1; then
    echo "Stopping training pid=${TRAIN_PID}"
    kill -TERM "${TRAIN_PID}" >/dev/null 2>&1 || true
    for _ in {1..10}; do
      kill -0 "${TRAIN_PID}" >/dev/null 2>&1 || break
      sleep 1
    done
    kill -0 "${TRAIN_PID}" >/dev/null 2>&1 && kill -KILL "${TRAIN_PID}" >/dev/null 2>&1 || true
    wait "${TRAIN_PID}" >/dev/null 2>&1 || true
  fi
}

cleanup_all() {
  local status=$?
  trap - EXIT INT TERM
  if [[ "${CLEANING_UP}" == "1" ]]; then
    return "${status}"
  fi
  CLEANING_UP=1
  cleanup_training
  cleanup_rollout
  return "${status}"
}

handle_interrupt() {
  cleanup_all
  exit 130
}

handle_terminate() {
  cleanup_all
  exit 143
}

trap cleanup_all EXIT
trap handle_interrupt INT
trap handle_terminate TERM

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"
CONDA_ENV_PREFIX="${CONDA_PREFIX:-/root/sunboyuan/miniconda3/envs/qwen3.5}"
export CUDA_HOME="${CUDA_HOME:-${CONDA_ENV_PREFIX}}"
export PATH="${CUDA_HOME}/bin:${PATH}"
CURAND_LIB_DIR="${CONDA_ENV_PREFIX}/lib/python3.10/site-packages/nvidia/curand/lib"
if [[ -d "${CURAND_LIB_DIR}" ]]; then
  export LIBRARY_PATH="${CURAND_LIB_DIR}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  export LD_LIBRARY_PATH="${CURAND_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

echo "Starting rollout server on ${ROLLOUT_HOST}:${ROLLOUT_PORT} with CUDA_VISIBLE_DEVICES=${ROLLOUT_CUDA_VISIBLE_DEVICES}"
echo "Rollout vLLM config: gpu_memory_utilization=${ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION}, max_model_len=${ROLLOUT_VLLM_MAX_MODEL_LEN}, max_num_seqs=${ROLLOUT_VLLM_MAX_NUM_SEQS}"
echo "Qwen explicit thinking: ${QWEN_ENABLE_THINKING}"
echo "Qwen response prefix: ${QWEN_RESPONSE_PREFIX:-<empty>}"
setsid env CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES}" swift rollout \
  --model "${MODEL}" \
  --enable_thinking "${QWEN_ENABLE_THINKING}" \
  --response_prefix "${QWEN_RESPONSE_PREFIX}" \
  --vllm_tensor_parallel_size "${ROLLOUT_TENSOR_PARALLEL_SIZE}" \
  --vllm_data_parallel_size "${ROLLOUT_DATA_PARALLEL_SIZE}" \
  --vllm_gpu_memory_utilization "${ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION}" \
  --vllm_max_model_len "${ROLLOUT_VLLM_MAX_MODEL_LEN}" \
  --vllm_max_num_seqs "${ROLLOUT_VLLM_MAX_NUM_SEQS}" \
  --port "${ROLLOUT_PORT}" \
  >"${ROLLOUT_LOG}" 2>&1 &
ROLLOUT_PID=$!
ROLLOUT_PGID="$(ps -o pgid= -p "${ROLLOUT_PID}" | tr -d ' ')"
start_rollout_monitor

echo "Waiting for rollout server to become ready. Log: ${ROLLOUT_LOG}"
for ((i = 1; i <= ROLLOUT_WAIT_SECONDS; i++)); do
  if ! kill -0 "${ROLLOUT_PID}" >/dev/null 2>&1; then
    echo "Rollout server exited before becoming ready. Last log lines:"
    tail -n 80 "${ROLLOUT_LOG}" || true
    exit 1
  fi
  if rollout_ready; then
    echo "Rollout server is ready."
    break
  fi
  sync_rollout_port_from_log
  if rollout_ready; then
    echo "Rollout server is ready."
    break
  fi
  if (( i == ROLLOUT_WAIT_SECONDS )); then
    echo "Timed out waiting for rollout server after ${ROLLOUT_WAIT_SECONDS}s. Last log lines:"
    tail -n 80 "${ROLLOUT_LOG}" || true
    exit 1
  fi
  sleep 1
done

echo "Starting training. Log: ${TRAIN_LOG}"
export TRAIN_LOG
setsid bash -c 'set -o pipefail; PYTHONUNBUFFERED=1 python "$@" 2>&1 | tee "${TRAIN_LOG}"' bash tools/train_geogrpo.py \
  --run-name "${RUN_NAME}" \
  --input /nfs/sunboyuan/Geobench/dataset/data_train/all/all_selected_merged.jsonl \
  --model "${MODEL}" \
  --system-prompt-file "${SYSTEM_PROMPT_FILE}" \
  --schedule four_stage \
  --domain-balance ratio \
  --domain-ratio ground:3,map:3,remote:3,street:3,indoor:1,landmark:1,roadnet:1,shape:1,space:1,uav:1 \
  --cuda-visible-devices "${TRAIN_CUDA_VISIBLE_DEVICES}" \
  --nproc-per-node "${TRAIN_NPROC_PER_NODE}" \
  --image-max-token-num none \
  --num-train-epochs 1 \
  --num-generations 8 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-5 \
  --max-length 10240 \
  --max-completion-length 4096 \
  --max-pixels 401408 \
  --deepspeed zero3 \
  --enable-thinking "${QWEN_ENABLE_THINKING}" \
  --response-prefix "${QWEN_RESPONSE_PREFIX}" \
  --vllm-mode server \
  --vllm-server-host "${ROLLOUT_HOST}" \
  --vllm-server-port "${ROLLOUT_PORT}" \
  --offload-optimizer false \
  --offload-model false \
  --freeze-vit true \
  --freeze-aligner true \
  --save-strategy steps \
  --save-steps 100 \
  --logging-steps 1 \
  --warmup-ratio 0.01 \
  --dataloader-num-workers 16 \
  --dataset-num-proc 16 \
  --temperature 0.9 \
  --gradient-checkpointing false \
  --vit-gradient-checkpointing false \
  --beta 0.001 \
  --num-iterations 1 \
  --swanlab-project Geobench \
  --swanlab-exp-name "${RUN_NAME}" \
  --reward-funcs format geoscore_accuracy geo_format \
  --geoscore-api-keys "${GEOSCORE_API_KEY_ARGS[@]}" \
  --geoscore-max-distance "${GEOSCORE_MAX_DISTANCE}" \
  --reward-weights 1.0 1.0 1.0 \
  &
TRAIN_PID=$!
TRAIN_PGID="$(ps -o pgid= -p "${TRAIN_PID}" | tr -d ' ')"
wait "${TRAIN_PID}"
