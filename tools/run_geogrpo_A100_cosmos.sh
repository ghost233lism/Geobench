#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/nas/zhangyiming/geobench}"
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
export REPO_ROOT
RUNTIME_DIR="${RUNTIME_DIR:-${REPO_ROOT}/tools/runtime}"
GEOBENCH_RUNTIME_ROOT="${GEOBENCH_RUNTIME_ROOT:-${REPO_ROOT}/output/runtime}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-cosmos}"
CONDA_CLONE_SOURCE="${CONDA_CLONE_SOURCE:-last05}"
REQUIRE_QWEN35_RUNTIME="${REQUIRE_QWEN35_RUNTIME:-false}"
SKIP_RUNTIME_ENV_CHECK="${SKIP_RUNTIME_ENV_CHECK:-false}"
QWEN35_TRANSFORMERS_VERSION="${QWEN35_TRANSFORMERS_VERSION:-5.2.0}"
QWEN35_VLLM_VERSION="${QWEN35_VLLM_VERSION:-0.17.1}"
WHEEL_DIR="${WHEEL_DIR:-${REPO_ROOT}/wheels}"
RUN_NAME="${RUN_NAME:-geogrpo_cosmos_smoke}"
MODEL="${MODEL:-/mnt/nas/zhangyiming/database/ckpt/pretrained/Qwen2.5-VL-7B-Instruct}"
ROLLOUT_MODEL="${ROLLOUT_MODEL:-${GEOBENCH_RUNTIME_ROOT}/models/$(basename "${MODEL}")-vllm-compat}"
TRAIN_MODEL="${TRAIN_MODEL:-${MODEL}}"
INPUT_JSONL="${INPUT_JSONL:-${REPO_ROOT}/all_selected_merged_current_paths.jsonl}"
QWEN35_NO_TAGS="${QWEN35_NO_TAGS:-false}"
case "${QWEN35_NO_TAGS,,}" in
  1|true|yes|on) qwen35_no_tags_enabled=true ;;
  *) qwen35_no_tags_enabled=false ;;
esac
if [[ "${qwen35_no_tags_enabled}" == "true" ]]; then
  SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-${REPO_ROOT}/tools/system_prompt_compact_no_think.txt}"
  QWEN35_USER_PROMPT="${QWEN35_USER_PROMPT:-<image> Based on the image, reason carefully about the visual clues and tell me the specific location. Output only the required JSON.}"
else
  SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-${REPO_ROOT}/tools/system_prompt.txt}"
  QWEN35_USER_PROMPT="${QWEN35_USER_PROMPT:-<image> Based on the image, tell me the specific location and your thinking process. Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.}"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/scheduled/${RUN_NAME}}"
CHECKPOINT_OUTPUT_ROOT="${CHECKPOINT_OUTPUT_ROOT:-/mnt/data/zhangyiming/database/ckpt/geobench}"
SWIFT_OUTPUT_DIR="${SWIFT_OUTPUT_DIR:-${CHECKPOINT_OUTPUT_ROOT}/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/output/${RUN_NAME}/logs}"
ROLLOUT_LOG="${ROLLOUT_LOG:-${LOG_DIR}/rollout.log}"
TRAIN_LOG="${TRAIN_LOG:-${LOG_DIR}/train.log}"
CHECKPOINT_OFFLOAD_LOG="${CHECKPOINT_OFFLOAD_LOG:-${LOG_DIR}/checkpoint-offload.log}"
OFFLOAD_CHECKPOINTS="${OFFLOAD_CHECKPOINTS:-false}"
if [[ "${OFFLOAD_CHECKPOINTS}" == "true" && "${SWIFT_OUTPUT_DIR%/}" == "${CHECKPOINT_OUTPUT_ROOT%/}/${RUN_NAME}" ]]; then
  echo "SWIFT_OUTPUT_DIR already points at the checkpoint data root; disabling checkpoint offload."
  OFFLOAD_CHECKPOINTS=false
fi

TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1}"
TRAIN_NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE:-2}"
VLLM_MODE="${VLLM_MODE:-server}"
ROLLOUT_CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES:-4,5}"
ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-2}"
ROLLOUT_DATA_PARALLEL_SIZE="${ROLLOUT_DATA_PARALLEL_SIZE:-1}"
ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION="${ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
ROLLOUT_VLLM_MAX_MODEL_LEN="${ROLLOUT_VLLM_MAX_MODEL_LEN:-3072}"
ROLLOUT_VLLM_MAX_NUM_SEQS="${ROLLOUT_VLLM_MAX_NUM_SEQS:-4}"
ROLLOUT_VLLM_ENFORCE_EAGER="${ROLLOUT_VLLM_ENFORCE_EAGER:-true}"
DEFAULT_ROLLOUT_VLLM_LIMIT_MM_PER_PROMPT='{"image":1}'
ROLLOUT_VLLM_LIMIT_MM_PER_PROMPT="${ROLLOUT_VLLM_LIMIT_MM_PER_PROMPT:-${DEFAULT_ROLLOUT_VLLM_LIMIT_MM_PER_PROMPT}}"
ROLLOUT_HOST="${ROLLOUT_HOST:-127.0.0.1}"
ROLLOUT_PORT="${ROLLOUT_PORT:-auto}"
ROLLOUT_WAIT_SECONDS="${ROLLOUT_WAIT_SECONDS:-900}"
MIN_ROLLOUT_GPU_FREE_MB="${MIN_ROLLOUT_GPU_FREE_MB:-20000}"
TRAIN_DONE_GRACE_SECONDS="${TRAIN_DONE_GRACE_SECONDS:-20}"
ROLLOUT_CLEANUP_GRACE_SECONDS="${ROLLOUT_CLEANUP_GRACE_SECONDS:-10}"

QWEN_ENABLE_THINKING="${QWEN_ENABLE_THINKING:-false}"
QWEN_RESPONSE_PREFIX="${QWEN_RESPONSE_PREFIX:-}"
ROLLOUT_TEMPLATE="${ROLLOUT_TEMPLATE:-qwen2_5_vl}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
TRAIN_DATA_RATIO="${TRAIN_DATA_RATIO:-1.0}"
SCHEDULE="${SCHEDULE:-four_stage}"
DOMAIN_BALANCE="${DOMAIN_BALANCE:-ratio}"
DOMAIN_RATIO_TEXT="${DOMAIN_RATIO_TEXT:-}"
DOMAIN_ORDER_TEXT="${DOMAIN_ORDER_TEXT:-}"
NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_LENGTH="${MAX_LENGTH:-3072}"
if [[ "${qwen35_no_tags_enabled}" == "true" ]]; then
  MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}"
else
  MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-768}"
fi
MAX_PIXELS="${MAX_PIXELS:-262144}"
IMAGE_MIN_TOKEN_NUM="${IMAGE_MIN_TOKEN_NUM:-4}"
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-512}"
DEEPSPEED="${DEEPSPEED:-zero3}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-2}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
VIT_GRADIENT_CHECKPOINTING="${VIT_GRADIENT_CHECKPOINTING:-false}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-20}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
BETA="${BETA:-0.04}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-2.0}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-${ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION}}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-${ROLLOUT_TENSOR_PARALLEL_SIZE}}"
MOVE_MODEL_BATCHES="${MOVE_MODEL_BATCHES:-8}"
SLEEP_LEVEL="${SLEEP_LEVEL:-1}"
if [[ "${qwen35_no_tags_enabled}" == "true" ]]; then
  REWARD_FUNCS_TEXT="${REWARD_FUNCS_TEXT:-geoscore_accuracy formatstrict}"
  REWARD_WEIGHTS_TEXT="${REWARD_WEIGHTS_TEXT:-2.0 1.0}"
else
  REWARD_FUNCS_TEXT="${REWARD_FUNCS_TEXT:-geoscore_accuracy geo_strict_format geo_no_unknown}"
  REWARD_WEIGHTS_TEXT="${REWARD_WEIGHTS_TEXT:-2.0 1.0 1.0}"
fi
GEOSCORE_API_KEY_CONFIG="${GEOSCORE_API_KEY_CONFIG:-${REPO_ROOT}/tools/geoscore_api_keys.conf}"
if [[ -f "${GEOSCORE_API_KEY_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${GEOSCORE_API_KEY_CONFIG}"
fi
GEOSCORE_MAX_DISTANCE="${GEOSCORE_MAX_DISTANCE:-18050.0}"
GEOSCORE_CACHE_FILE="${GEOSCORE_CACHE_FILE:-${LOG_DIR}/geoscore_cache.json}"
GEOSCORE_API_KEY_ARGS=()
if [[ -n "${GEOSCORE_API_KEYS:-}" ]]; then
  export GEOSCORE_API_KEYS
  read -r -a GEOSCORE_API_KEY_ARGS <<< "${GEOSCORE_API_KEYS//,/ }"
fi
export GEOSCORE_CACHE_FILE

# shellcheck source=/dev/null
source "${REPO_ROOT}/tools/tracking_env.sh"
geobench_configure_tracking "${RUN_NAME}"
EXTRA_SWIFT_ARGS_TEXT="${EXTRA_SWIFT_ARGS_TEXT:---report_to ${REPORT_TO}}"

if [[ -f /mnt/nas/zhangyiming/clash/env.sh ]]; then
  # shellcheck source=/dev/null
  source /mnt/nas/zhangyiming/clash/env.sh auto >/dev/null 2>&1 || true
fi
if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/etc/profile.d/conda.sh
fi
export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"
export PYTHONPATH="${RUNTIME_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

env_exists() {
  conda env list | awk '{print $1}' | grep -qx "$1"
}

cleanup_stale_runtime_processes() {
  local patterns=(
    "/mnt/nas/zhangyiming/train/bash/cosmos_cot.py"
    "${GEOBENCH_RUNTIME_ROOT}/models/.*-vllm-compat"
    "Qwen3.5-9B-vllm-compat"
    "Qwen2.5-VL-7B-Instruct-vllm-compat"
    "vllm.entrypoints"
    "${REPO_ROOT}/swift/cli/rollout.py"
    "${REPO_ROOT}/swift/cli/rlhf.py --rlhf_type grpo"
    "from multiprocessing.spawn import spawn_main"
    "from multiprocessing.resource_tracker import main"
  )

  echo "Cleaning stale GeoBench/default GPU processes if any."
  for pattern in "${patterns[@]}"; do
    pkill -TERM -f "${pattern}" >/dev/null 2>&1 || true
  done
  sleep 3
  for pattern in "${patterns[@]}"; do
    pkill -KILL -f "${pattern}" >/dev/null 2>&1 || true
  done
}

wait_for_rollout_gpu_memory() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local -a gpu_ids
  IFS=',' read -r -a gpu_ids <<< "${ROLLOUT_CUDA_VISIBLE_DEVICES}"

  for _ in {1..30}; do
    local all_ready=1
    local status=""
    local gpu_id free_mb
    for gpu_id in "${gpu_ids[@]}"; do
      free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu_id}" 2>/dev/null | awk 'NR==1 {gsub(/ /, ""); print $1}')"
      status+="${gpu_id}:${free_mb:-unknown}MB "
      if [[ -z "${free_mb}" || "${free_mb}" -lt "${MIN_ROLLOUT_GPU_FREE_MB}" ]]; then
        all_ready=0
      fi
    done
    echo "Rollout GPU free memory: ${status}(required >= ${MIN_ROLLOUT_GPU_FREE_MB}MB each)"
    if [[ "${all_ready}" -eq 1 ]]; then
      return 0
    fi
    cleanup_stale_runtime_processes
    sleep 2
  done

  echo "Rollout GPUs did not free enough memory after stale-process cleanup." >&2
  nvidia-smi >&2 || true
  return 1
}

cleanup_stale_runtime_processes

ensure_qwen35_runtime() {
  # The source cosmos env may contain flash-attn compiled against torch 2.6.
  # Qwen3.5 uses a torch 2.10/vLLM 0.17 stack here, so remove the stale wheel
  # and force Swift/Transformers to use the SDPA/eager path instead.
  python -m pip uninstall -y flash-attn flash_attn >/dev/null 2>&1 || true
  if python - "${MODEL}" "${QWEN35_TRANSFORMERS_VERSION}" "${QWEN35_VLLM_VERSION}" <<'PY'
import importlib
import importlib.metadata as md
import os
import sys
from pathlib import Path

from packaging.version import Version

model_dir, min_transformers, min_vllm = sys.argv[1:4]
repo_root = Path(os.environ['REPO_ROOT']).resolve()

for name in ('torch', 'triton', 'transformers', 'vllm', 'gradio', 'nltk', 'trl', 'swift',
             'qwen_vl_utils', 'fla', 'tilelang', 'wandb'):
    module = importlib.import_module(name)
    if name == 'swift':
        swift_path = Path(module.__file__).resolve()
        if not str(swift_path).startswith(str(repo_root)):
            raise RuntimeError(f'swift is not loaded from local repo: {swift_path}')
from trl.trainer.grpo_trainer import GRPOTrainer  # noqa: F401
if Version(md.version('transformers')) < Version(min_transformers):
    raise RuntimeError(f'transformers is too old: {md.version("transformers")}')
if Version(md.version('vllm')) < Version(min_vllm):
    raise RuntimeError(f'vllm is too old: {md.version("vllm")}')
if int(md.version('gradio').split('.', 1)[0]) >= 6:
    raise RuntimeError(f'gradio is outside ms-swift range: {md.version("gradio")}')
if tuple(int(part) for part in md.version('trl').split('.')[:2]) < (0, 20):
    raise RuntimeError(f'trl is below ms-swift RLHF requirement: {md.version("trl")}')

from transformers import AutoConfig, AutoProcessor

config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
if getattr(config, 'model_type', None) != 'qwen3_5':
    raise RuntimeError(f'expected qwen3_5 model_type, got {getattr(config, "model_type", None)}')
AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
PY
  then
    echo "Conda env ${CONDA_ENV_NAME} already has the required Qwen3.5 runtime imports."
    return 0
  fi

  echo "Installing Qwen3.5 runtime packages into ${CONDA_ENV_NAME}."
  # Qwen3.5 support needs a newer vLLM stack than the cloned 7B cosmos env.
  # Install it only inside the Qwen3.5 env; vLLM 0.17.1 pins torch 2.10.0
  # with CUDA 12.8 wheels, which avoids the ABI mismatch seen with torch 2.6.
  python -m pip uninstall -y vllm >/dev/null 2>&1 || true
  python -m pip install -U "vllm==${QWEN35_VLLM_VERSION}"
  # vLLM pins transformers <5, while Qwen3.5 training/model loading requires
  # transformers 5.2.0. Reassert it after vLLM installs its dependency stack.
  python -m pip install -U --no-deps \
    "huggingface-hub>=1.3.0,<2.0" \
    "tokenizers>=0.22.0,<=0.23.0" \
    "typer-slim" \
    "transformers==${QWEN35_TRANSFORMERS_VERSION}"
  python -m pip install -U "tilelang"
  python -m pip install -U "wandb>=0.22.2"

  python - "${MODEL}" <<'PY'
import importlib
import sys
from transformers import AutoConfig, AutoProcessor

for name in ('torch', 'transformers', 'vllm', 'qwen_vl_utils', 'fla', 'tilelang', 'swift', 'wandb'):
    importlib.import_module(name)
from trl.trainer.grpo_trainer import GRPOTrainer  # noqa: F401
config = AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True)
assert getattr(config, 'model_type', None) == 'qwen3_5', getattr(config, 'model_type', None)
AutoProcessor.from_pretrained(sys.argv[1], trust_remote_code=True)
print('Qwen3.5 runtime import check passed.')
PY
}

bootstrap_cosmos_env() {
  if ! env_exists "${CONDA_ENV_NAME}"; then
    echo "Conda env ${CONDA_ENV_NAME} not found on this runner; cloning ${CONDA_CLONE_SOURCE}."
    if ! env_exists "${CONDA_CLONE_SOURCE}"; then
      echo "Clone source conda env ${CONDA_CLONE_SOURCE} is not available." >&2
      conda env list >&2
      exit 1
    fi
    CONDA_NO_PLUGINS=true conda create --offline -n "${CONDA_ENV_NAME}" --clone "${CONDA_CLONE_SOURCE}" -y
  fi

  conda activate "${CONDA_ENV_NAME}"
  python -m pip install -e "${REPO_ROOT}" --no-deps --no-build-isolation

  if [[ "${SKIP_RUNTIME_ENV_CHECK}" == "true" || "${SKIP_RUNTIME_ENV_CHECK}" == "1" ]]; then
    echo "Skipping runtime import checks and package installation."
    return 0
  fi

  if [[ "${REQUIRE_QWEN35_RUNTIME}" == "true" || "${REQUIRE_QWEN35_RUNTIME}" == "1" ]]; then
    ensure_qwen35_runtime
    python -m pip install -e "${REPO_ROOT}" --no-deps --no-build-isolation
    echo "GeoBench Qwen3.5 runtime is ready."
    return 0
  fi

  if python - <<'PY'
import importlib
import os
from importlib.metadata import version
from pathlib import Path

repo_root = Path(os.environ['REPO_ROOT']).resolve()

for name in ('torch', 'triton', 'flash_attn', 'transformers', 'vllm', 'gradio', 'nltk', 'trl', 'swift'):
    module = importlib.import_module(name)
    if name == 'swift':
        swift_path = Path(module.__file__).resolve()
        if not str(swift_path).startswith(str(repo_root)):
            raise RuntimeError(f'swift is not loaded from local repo: {swift_path}')
from trl import GRPOTrainer  # noqa: F401
if int(version('gradio').split('.', 1)[0]) >= 6:
    raise RuntimeError(f'gradio is outside ms-swift range: {version("gradio")}')
if tuple(int(part) for part in version('trl').split('.')[:2]) < (0, 20):
    raise RuntimeError(f'trl is below ms-swift RLHF requirement: {version("trl")}')
PY
  then
    echo "Conda env ${CONDA_ENV_NAME} already has the required GeoBench runtime imports."
    return 0
  fi

  echo "Installing GeoBench runtime packages into ${CONDA_ENV_NAME}."
  python -m pip install "qwen_vl_utils>=0.0.14" peft==0.18.1 "datasets<4.0" deepspeed "gradio>=3.40.0,<5.33" nltk "trl>=0.20,<0.21" mergekit
  python -m pip install modelscope binpacking cpm-kernels dacite json-repair openai oss2 rouge \
    sentencepiece simplejson sortedcontainers tensorboard tiktoken transformers-stream-generator zstandard pyecharts
  python -m pip install absl-py markdown tensorboard-data-server \
    aliyun-python-sdk-core aliyun-python-sdk-kms crcmod pycryptodome toml gym-notices
  local optional_wheels=()
  for wheel in \
    "${WHEEL_DIR}/liger_kernel-0.8.0-py3-none-any.whl" \
    "${WHEEL_DIR}/swanlab-0.7.19-py3-none-any.whl" \
    "${WHEEL_DIR}/flash_linear_attention-0.5.0-py3-none-any.whl" \
    "${WHEEL_DIR}/fla_core-0.5.0-py3-none-any.whl"; do
    [[ -f "${wheel}" ]] && optional_wheels+=("${wheel}")
  done
  if (( ${#optional_wheels[@]} )); then
    python -m pip install --no-deps "${optional_wheels[@]}" || true
  fi
  local xformers_wheel="${WHEEL_DIR}/xformers-0.0.29.post2-cp310-cp310-manylinux_2_28_x86_64.whl"
  local vllm_wheel="${WHEEL_DIR}/vllm-0.8.5.post1-cp38-abi3-manylinux1_x86_64.whl"
  if [[ -f "${xformers_wheel}" && -f "${vllm_wheel}" ]]; then
    python -m pip install --find-links "${WHEEL_DIR}" "${xformers_wheel}" "${vllm_wheel}"
  else
    python -m pip install xformers==0.0.29.post2 vllm==0.8.5.post1
  fi
  if ! python - <<'PY'
import triton
raise SystemExit(0 if triton.__version__ == '3.2.0' else 1)
PY
  then
    python -m pip install --force-reinstall --no-deps triton==3.2.0
  fi
  rm -rf "${CONDA_PREFIX}/lib/python3.10/site-packages/~riton" \
    "${CONDA_PREFIX}/lib/python3.10/site-packages/~riton-3.2.0.dist-info"

  python - <<'PY'
import importlib

for name in ('torch', 'triton', 'flash_attn', 'transformers', 'vllm', 'gradio', 'nltk', 'trl', 'swift'):
    importlib.import_module(name)
from trl import GRPOTrainer  # noqa: F401
print('GeoBench runtime import check passed.')
PY
}

bootstrap_cosmos_env

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}" "${SWIFT_OUTPUT_DIR}"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"
export GEOBENCH_QWEN2VL_MIN_PIXELS="${GEOBENCH_QWEN2VL_MIN_PIXELS:-3136}"
export GEOBENCH_QWEN2VL_MAX_PIXELS="${GEOBENCH_QWEN2VL_MAX_PIXELS:-${MAX_PIXELS}}"
export IMAGE_MIN_TOKEN_NUM
export IMAGE_MAX_TOKEN_NUM
export min_pixels="${GEOBENCH_QWEN2VL_MIN_PIXELS}"
export max_pixels="${GEOBENCH_QWEN2VL_MAX_PIXELS}"

CONDA_ENV_PREFIX="${CONDA_PREFIX:-/root/miniconda3/envs/cosmos}"
if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  for cuda_candidate in /usr/local/cuda /usr/local/cuda-12.4 /usr/local/cuda-12 /usr/local/cuda-*; do
    if [[ -x "${cuda_candidate}/bin/nvcc" ]]; then
      export CUDA_HOME="${cuda_candidate}"
      break
    fi
  done
fi
export CUDA_HOME="${CUDA_HOME:-${CONDA_ENV_PREFIX}}"
export PATH="${CUDA_HOME}/bin:${PATH}"
CURAND_LIB_DIR="${CONDA_ENV_PREFIX}/lib/python3.10/site-packages/nvidia/curand/lib"
if [[ -d "${CURAND_LIB_DIR}" ]]; then
  export LIBRARY_PATH="${CURAND_LIB_DIR}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  export LD_LIBRARY_PATH="${CURAND_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

read -r -a REWARD_FUNCS <<< "${REWARD_FUNCS_TEXT}"
read -r -a REWARD_WEIGHTS <<< "${REWARD_WEIGHTS_TEXT}"
read -r -a EXTRA_SWIFT_ARGS <<< "${EXTRA_SWIFT_ARGS_TEXT}"
MAX_SAMPLES_ARGS=()
if [[ -n "${MAX_SAMPLES}" && "${MAX_SAMPLES}" != "0" && "${MAX_SAMPLES}" != "all" ]]; then
  MAX_SAMPLES_ARGS=(--max-samples "${MAX_SAMPLES}")
fi
TRAIN_DATA_RATIO_ARGS=()
if [[ -n "${TRAIN_DATA_RATIO}" && "${TRAIN_DATA_RATIO}" != "1" && "${TRAIN_DATA_RATIO}" != "1.0" && "${TRAIN_DATA_RATIO}" != "100%" && "${TRAIN_DATA_RATIO}" != "all" ]]; then
  TRAIN_DATA_RATIO_ARGS=(--train-data-ratio "${TRAIN_DATA_RATIO}")
fi
DOMAIN_RATIO_ARGS=()
if [[ -n "${DOMAIN_RATIO_TEXT}" ]]; then
  DOMAIN_RATIO_ARGS=(--domain-ratio "${DOMAIN_RATIO_TEXT}")
fi
DOMAIN_ORDER_ARGS=()
if [[ -n "${DOMAIN_ORDER_TEXT}" ]]; then
  DOMAIN_ORDER_ARGS=(--domain-order "${DOMAIN_ORDER_TEXT}")
fi

prepare_vllm_compat_model() {
  python - "${MODEL}" "${ROLLOUT_MODEL}" <<'PY'
import json
import os
import shutil
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2]).resolve()
target.mkdir(parents=True, exist_ok=True)

for item in source.iterdir():
    if item.name == 'config.json':
        continue
    link = target / item.name
    if link.is_symlink() and Path(os.readlink(link)) == item:
        continue
    if link.exists() or link.is_symlink():
        if link.is_dir() and not link.is_symlink():
            shutil.rmtree(link)
        else:
            link.unlink()
    os.symlink(item, link)

config = json.loads((source / 'config.json').read_text())
rope_scaling = config.get('rope_scaling')
if isinstance(rope_scaling, dict) and rope_scaling.get('type') == 'mrope':
    rope_scaling = dict(rope_scaling)
    rope_scaling.pop('type', None)
    rope_scaling['rope_type'] = 'mrope'
    config['rope_scaling'] = rope_scaling

config_path = target / 'config.json'
if config_path.is_symlink() or config_path.exists():
    config_path.unlink()
config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n')
print(target)
PY
}

choose_rollout_port() {
  if [[ "${ROLLOUT_PORT}" == "auto" ]]; then
    ROLLOUT_PORT="$(
      PYTHONPATH="" python - "${ROLLOUT_HOST}" <<'PY'
import socket
import sys

host = sys.argv[1]
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind((host, 0))
    print(sock.getsockname()[1])
PY
    )"
  fi
}

rollout_ready() {
  local url="http://${ROLLOUT_HOST}:${ROLLOUT_PORT}/health/"
  curl -fsS "${url}" >/dev/null 2>&1
}

sync_rollout_port_from_log() {
  local actual_port
  actual_port="$(
    PYTHONPATH="" python - "${ROLLOUT_LOG}" <<'PY' 2>/dev/null || true
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
if log_path.exists():
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
  if [[ -z "${ROLLOUT_PGID:-}" && -z "${ROLLOUT_PID:-}" ]]; then
    return 0
  fi

  local self_pgid=""
  self_pgid="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d ' ' || true)"
  local can_kill_group=0
  if [[ -n "${ROLLOUT_PGID:-}" && "${ROLLOUT_PGID}" != "${self_pgid}" ]] \
      && kill -0 -- "-${ROLLOUT_PGID}" >/dev/null 2>&1; then
    can_kill_group=1
  fi

  echo "Stopping rollout process group pgid=${ROLLOUT_PGID:-unknown} pid=${ROLLOUT_PID:-unknown}"
  if [[ "${can_kill_group}" -eq 1 ]]; then
    kill -TERM -- "-${ROLLOUT_PGID}" >/dev/null 2>&1 || true
  elif [[ -n "${ROLLOUT_PID:-}" ]]; then
    kill -TERM "${ROLLOUT_PID}" >/dev/null 2>&1 || true
  fi

  for ((i = 0; i < ROLLOUT_CLEANUP_GRACE_SECONDS; i++)); do
    if [[ "${can_kill_group}" -eq 1 ]]; then
      kill -0 -- "-${ROLLOUT_PGID}" >/dev/null 2>&1 || {
        echo "Rollout process group stopped."
        return 0
      }
    elif [[ -n "${ROLLOUT_PID:-}" ]]; then
      kill -0 "${ROLLOUT_PID}" >/dev/null 2>&1 || {
        echo "Rollout process stopped."
        return 0
      }
    else
      echo "Rollout cleanup complete."
      return 0
    fi
    sleep 1
  done

  echo "Rollout still alive after ${ROLLOUT_CLEANUP_GRACE_SECONDS}s; forcing kill."
  if [[ "${can_kill_group}" -eq 1 ]]; then
    kill -KILL -- "-${ROLLOUT_PGID}" >/dev/null 2>&1 || true
  elif [[ -n "${ROLLOUT_PID:-}" ]]; then
    kill -KILL "${ROLLOUT_PID}" >/dev/null 2>&1 || true
  fi

  for _ in {1..5}; do
    if [[ "${can_kill_group}" -eq 1 ]]; then
      kill -0 -- "-${ROLLOUT_PGID}" >/dev/null 2>&1 || break
    elif [[ -n "${ROLLOUT_PID:-}" ]]; then
      kill -0 "${ROLLOUT_PID}" >/dev/null 2>&1 || break
    else
      break
    fi
    sleep 1
  done
  echo "Rollout cleanup complete."
}

cleanup_training() {
  local self_pgid=""
  self_pgid="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d ' ' || true)"
  if [[ -n "${TRAIN_PGID:-}" && "${TRAIN_PGID}" != "${self_pgid}" ]] \
      && kill -0 -- "-${TRAIN_PGID}" >/dev/null 2>&1; then
    echo "Stopping training process group pgid=${TRAIN_PGID}"
    kill -TERM -- "-${TRAIN_PGID}" >/dev/null 2>&1 || true
    for _ in {1..10}; do
      kill -0 -- "-${TRAIN_PGID}" >/dev/null 2>&1 || break
      sleep 1
    done
    kill -0 -- "-${TRAIN_PGID}" >/dev/null 2>&1 && kill -KILL -- "-${TRAIN_PGID}" >/dev/null 2>&1 || true
  elif [[ -n "${TRAIN_PID:-}" ]] && kill -0 "${TRAIN_PID}" >/dev/null 2>&1; then
    echo "Stopping training process pid=${TRAIN_PID}"
    kill -TERM "${TRAIN_PID}" >/dev/null 2>&1 || true
    sleep 3
    kill -0 "${TRAIN_PID}" >/dev/null 2>&1 && kill -KILL "${TRAIN_PID}" >/dev/null 2>&1 || true
  fi
}

CHECKPOINT_OFFLOADER_PID=""

start_checkpoint_offloader() {
  if [[ "${OFFLOAD_CHECKPOINTS}" != "true" ]]; then
    echo "Checkpoint offload disabled."
    return 0
  fi
  : >"${CHECKPOINT_OFFLOAD_LOG}"
  bash "${REPO_ROOT}/tools/offload_checkpoints.sh" \
    "${SWIFT_OUTPUT_DIR}" \
    "${CHECKPOINT_OUTPUT_ROOT}" \
    "${RUN_NAME}" \
    "${SAVE_TOTAL_LIMIT}" \
    >>"${CHECKPOINT_OFFLOAD_LOG}" 2>&1 &
  CHECKPOINT_OFFLOADER_PID="$!"
  echo "Checkpoint offloader log: ${CHECKPOINT_OFFLOAD_LOG}"
  echo "Checkpoint target root: ${CHECKPOINT_OUTPUT_ROOT}/${RUN_NAME}"
}

run_checkpoint_offload_once() {
  if [[ "${OFFLOAD_CHECKPOINTS}" != "true" ]]; then
    return 0
  fi
  CHECKPOINT_OFFLOAD_ONCE=true bash "${REPO_ROOT}/tools/offload_checkpoints.sh" \
    "${SWIFT_OUTPUT_DIR}" \
    "${CHECKPOINT_OUTPUT_ROOT}" \
    "${RUN_NAME}" \
    "${SAVE_TOTAL_LIMIT}" \
    >>"${CHECKPOINT_OFFLOAD_LOG}" 2>&1 || true
}

stop_checkpoint_offloader() {
  if [[ -n "${CHECKPOINT_OFFLOADER_PID}" ]]; then
    kill "${CHECKPOINT_OFFLOADER_PID}" >/dev/null 2>&1 || true
    wait "${CHECKPOINT_OFFLOADER_PID}" >/dev/null 2>&1 || true
    CHECKPOINT_OFFLOADER_PID=""
  fi
}

cleanup_all() {
  local status=$?
  trap - EXIT INT
  trap '' TERM
  cleanup_training
  stop_checkpoint_offloader
  cleanup_rollout
  trap - TERM
  return "${status}"
}

trap cleanup_all EXIT
trap 'cleanup_all; exit 130' INT
trap 'cleanup_all; exit 143' TERM

prepare_vllm_compat_model
if [[ "${VLLM_MODE}" == "server" ]]; then
  choose_rollout_port
else
  export GEOBENCH_COLOCATE_VLLM_MODEL_DIR="${GEOBENCH_COLOCATE_VLLM_MODEL_DIR:-${ROLLOUT_MODEL}}"
  ROLLOUT_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}"
fi
wait_for_rollout_gpu_memory
: >"${ROLLOUT_LOG}"
: >"${TRAIN_LOG}"

echo "Run name: ${RUN_NAME}"
echo "Repo: ${REPO_ROOT}"
echo "Model: ${TRAIN_MODEL}"
echo "Rollout model: ${ROLLOUT_MODEL}"
echo "Colocate vLLM model: ${GEOBENCH_COLOCATE_VLLM_MODEL_DIR:-<server-mode>}"
echo "Input: ${INPUT_JSONL}"
echo "Log dir: ${LOG_DIR}"
echo "Swift output/checkpoint dir: ${SWIFT_OUTPUT_DIR}"
echo "Checkpoint direct root: ${CHECKPOINT_OUTPUT_ROOT}/${RUN_NAME} (offload=${OFFLOAD_CHECKPOINTS})"
echo "Training GPUs: ${TRAIN_CUDA_VISIBLE_DEVICES}; rollout GPUs: ${ROLLOUT_CUDA_VISIBLE_DEVICES}; vllm_mode=${VLLM_MODE}"
echo "Rollout config: port=${ROLLOUT_PORT}, tp=${ROLLOUT_TENSOR_PARALLEL_SIZE}, gpu_memory=${ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION}, max_model_len=${ROLLOUT_VLLM_MAX_MODEL_LEN}, max_num_seqs=${ROLLOUT_VLLM_MAX_NUM_SEQS}, enforce_eager=${ROLLOUT_VLLM_ENFORCE_EAGER}, max_pixels=${GEOBENCH_QWEN2VL_MAX_PIXELS}, image_token_range=${IMAGE_MIN_TOKEN_NUM}-${IMAGE_MAX_TOKEN_NUM}"
echo "Colocate config: tp=${VLLM_TENSOR_PARALLEL_SIZE}, gpu_memory=${VLLM_GPU_MEMORY_UTILIZATION}, move_model_batches=${MOVE_MODEL_BATCHES}"
echo "Training config: num_generations=${NUM_GENERATIONS}, batch=${PER_DEVICE_TRAIN_BATCH_SIZE}, max_length=${MAX_LENGTH}, max_completion_length=${MAX_COMPLETION_LENGTH}, learning_rate=${LEARNING_RATE}, beta=${BETA}"
echo "Data selection: train_data_ratio=${TRAIN_DATA_RATIO}, max_samples=${MAX_SAMPLES:-all}, schedule=${SCHEDULE}, domain_balance=${DOMAIN_BALANCE}, domain_ratio=${DOMAIN_RATIO_TEXT:-default}, domain_order=${DOMAIN_ORDER_TEXT:-default}"
echo "Qwen3.5 no-tags JSON mode: ${QWEN35_NO_TAGS}; enable_thinking=${QWEN_ENABLE_THINKING}; system_prompt=${SYSTEM_PROMPT_FILE}; user_prompt=${QWEN35_USER_PROMPT}"
echo "Rewards: ${REWARD_FUNCS_TEXT}"
echo "GeoScore API keys configured: ${#GEOSCORE_API_KEY_ARGS[@]}; cache=${GEOSCORE_CACHE_FILE}; max_distance=${GEOSCORE_MAX_DISTANCE}"

if [[ "${VLLM_MODE}" == "server" ]]; then
  setsid env CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES}" swift rollout \
    --model "${ROLLOUT_MODEL}" \
    --template "${ROLLOUT_TEMPLATE}" \
    --enable_thinking "${QWEN_ENABLE_THINKING}" \
    --response_prefix "${QWEN_RESPONSE_PREFIX}" \
    --vllm_tensor_parallel_size "${ROLLOUT_TENSOR_PARALLEL_SIZE}" \
    --vllm_data_parallel_size "${ROLLOUT_DATA_PARALLEL_SIZE}" \
    --vllm_gpu_memory_utilization "${ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION}" \
    --vllm_max_model_len "${ROLLOUT_VLLM_MAX_MODEL_LEN}" \
    --vllm_max_num_seqs "${ROLLOUT_VLLM_MAX_NUM_SEQS}" \
    --vllm_enforce_eager "${ROLLOUT_VLLM_ENFORCE_EAGER}" \
    --vllm_limit_mm_per_prompt "${ROLLOUT_VLLM_LIMIT_MM_PER_PROMPT}" \
    --max_pixels "${MAX_PIXELS}" \
    --port "${ROLLOUT_PORT}" \
    >"${ROLLOUT_LOG}" 2>&1 &
  ROLLOUT_PID=$!
  ROLLOUT_PGID="$(ps -o pgid= -p "${ROLLOUT_PID}" | tr -d ' ')"

  echo "Waiting for rollout server. Log: ${ROLLOUT_LOG}"
  for ((i = 1; i <= ROLLOUT_WAIT_SECONDS; i++)); do
    if ! kill -0 "${ROLLOUT_PID}" >/dev/null 2>&1; then
      echo "Rollout server exited before readiness. Last log lines:"
      tail -n 120 "${ROLLOUT_LOG}" || true
      exit 1
    fi
    sync_rollout_port_from_log
    if grep -Eq 'Application startup failed|ValueError: Found conflicts between|Traceback \(most recent call last\)' "${ROLLOUT_LOG}" >/dev/null 2>&1; then
      echo "Rollout server reported startup failure. Last log lines:"
      tail -n 160 "${ROLLOUT_LOG}" || true
      exit 1
    fi
    if rollout_ready; then
      echo "Rollout server is ready."
      break
    fi
    if (( i == ROLLOUT_WAIT_SECONDS )); then
      echo "Timed out waiting for rollout server after ${ROLLOUT_WAIT_SECONDS}s. Last log lines:"
      tail -n 120 "${ROLLOUT_LOG}" || true
      exit 1
    fi
    sleep 1
  done
else
  echo "Colocate mode: no external rollout server." >"${ROLLOUT_LOG}"
fi

start_checkpoint_offloader

TRAIN_VLLM_ARGS=(--vllm-mode "${VLLM_MODE}")
if [[ "${VLLM_MODE}" == "server" ]]; then
  TRAIN_VLLM_ARGS+=(--vllm-server-host "${ROLLOUT_HOST}" --vllm-server-port "${ROLLOUT_PORT}")
else
  TRAIN_VLLM_ARGS+=(
    --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
    --vllm-tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}"
    --move-model-batches "${MOVE_MODEL_BATCHES}"
    --sleep-level "${SLEEP_LEVEL}"
  )
fi

GEOSCORE_ARGS=(--geoscore-max-distance "${GEOSCORE_MAX_DISTANCE}")
if [[ "${GEOSCORE_PASS_API_KEYS_ON_CLI:-false}" == "true" ]] && (( ${#GEOSCORE_API_KEY_ARGS[@]} > 0 )); then
  GEOSCORE_ARGS+=(--geoscore-api-keys "${GEOSCORE_API_KEY_ARGS[@]}")
fi

echo "Starting training. Log: ${TRAIN_LOG}"
export TRAIN_LOG
setsid bash -c 'set -o pipefail; python "$@" 2>&1 | tee "${TRAIN_LOG}"' bash tools/train_geogrpo.py \
  --run-name "${RUN_NAME}" \
  --input "${INPUT_JSONL}" \
  --model "${TRAIN_MODEL}" \
  --system-prompt-file "${SYSTEM_PROMPT_FILE}" \
  --user-prompt "${QWEN35_USER_PROMPT}" \
  --output-root "${OUTPUT_ROOT}" \
  --output-dir "${SWIFT_OUTPUT_DIR}" \
  "${TRAIN_DATA_RATIO_ARGS[@]}" \
  "${MAX_SAMPLES_ARGS[@]}" \
  --schedule "${SCHEDULE}" \
  --domain-balance "${DOMAIN_BALANCE}" \
  "${DOMAIN_RATIO_ARGS[@]}" \
  "${DOMAIN_ORDER_ARGS[@]}" \
  --cuda-visible-devices "${TRAIN_CUDA_VISIBLE_DEVICES}" \
  --nproc-per-node "${TRAIN_NPROC_PER_NODE}" \
  --image-max-token-num none \
  --num-train-epochs 1 \
  --num-generations "${NUM_GENERATIONS}" \
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning-rate "${LEARNING_RATE}" \
  --torch-dtype "${TORCH_DTYPE}" \
  --max-length "${MAX_LENGTH}" \
  --max-completion-length "${MAX_COMPLETION_LENGTH}" \
  --max-pixels "${MAX_PIXELS}" \
  --deepspeed "${DEEPSPEED}" \
  --enable-thinking "${QWEN_ENABLE_THINKING}" \
  --response-prefix "${QWEN_RESPONSE_PREFIX}" \
  "${TRAIN_VLLM_ARGS[@]}" \
  --offload-optimizer false \
  --offload-model false \
  --freeze-vit true \
  --freeze-aligner true \
  --save-strategy "${SAVE_STRATEGY}" \
  --save-steps "${SAVE_STEPS}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT}" \
  --logging-steps 1 \
  --warmup-ratio "${WARMUP_RATIO}" \
  --max-grad-norm "${MAX_GRAD_NORM}" \
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS}" \
  --dataset-num-proc "${DATASET_NUM_PROC}" \
  --temperature "${TEMPERATURE}" \
  --gradient-checkpointing "${GRADIENT_CHECKPOINTING}" \
  --vit-gradient-checkpointing "${VIT_GRADIENT_CHECKPOINTING}" \
  --beta "${BETA}" \
  --num-iterations 1 \
  --swanlab-project "${SWANLAB_PROJECT:-Geobench}" \
  --swanlab-exp-name "${SWANLAB_EXP_NAME:-${RUN_NAME}}" \
  --reward-funcs "${REWARD_FUNCS[@]}" \
  --reward-weights "${REWARD_WEIGHTS[@]}" \
  "${GEOSCORE_ARGS[@]}" \
  --extra-swift-args "${EXTRA_SWIFT_ARGS[@]}" &
TRAIN_PID=$!
TRAIN_PGID="$(ps -o pgid= -p "${TRAIN_PID}" | tr -d ' ')"
set +e
TRAIN_STATUS=""
while true; do
  if ! kill -0 "${TRAIN_PID}" >/dev/null 2>&1; then
    wait "${TRAIN_PID}"
    TRAIN_STATUS=$?
    break
  fi

  if grep -q "End time of running main" "${TRAIN_LOG}" 2>/dev/null; then
    echo "Training main completion marker detected; waiting up to ${TRAIN_DONE_GRACE_SECONDS}s for wrapper exit."
    for ((i = 0; i < TRAIN_DONE_GRACE_SECONDS; i++)); do
      if ! kill -0 "${TRAIN_PID}" >/dev/null 2>&1; then
        wait "${TRAIN_PID}"
        TRAIN_STATUS=$?
        break
      fi
      sleep 1
    done

    if [[ -n "${TRAIN_STATUS}" ]]; then
      break
    fi

    if kill -0 "${TRAIN_PID}" >/dev/null 2>&1; then
      echo "Training wrapper still alive after main completion; terminating train process group."
      if [[ -n "${TRAIN_PGID}" ]]; then
        kill -TERM -- "-${TRAIN_PGID}" >/dev/null 2>&1 || true
      else
        kill -TERM "${TRAIN_PID}" >/dev/null 2>&1 || true
      fi
      sleep 3
      if kill -0 "${TRAIN_PID}" >/dev/null 2>&1; then
        if [[ -n "${TRAIN_PGID}" ]]; then
          kill -KILL -- "-${TRAIN_PGID}" >/dev/null 2>&1 || true
        else
          kill -KILL "${TRAIN_PID}" >/dev/null 2>&1 || true
        fi
      fi
      wait "${TRAIN_PID}" >/dev/null 2>&1 || true
      TRAIN_STATUS=0
      break
    fi
  fi

  sleep 2
done
set -e
echo "Training process exited with status ${TRAIN_STATUS}"
run_checkpoint_offload_once
stop_checkpoint_offloader
exit "${TRAIN_STATUS}"
