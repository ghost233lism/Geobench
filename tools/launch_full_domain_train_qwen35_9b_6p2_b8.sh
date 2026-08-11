#!/usr/bin/env bash
set -euo pipefail

# Train all 10 GeoBench domains one by one from the base model.
# Each domain uses full data and the internal four-stage difficulty schedule.
# Set START_DOMAIN=remote to resume from a later domain without rerunning earlier ones.

REPO_ROOT="${REPO_ROOT:-/mnt/nas/zhangyiming/geobench}"
LAUNCHER="${LAUNCHER:-${REPO_ROOT}/tools/launch_geogrpo_qwen35_9b_6p2_b8.sh}"
SOURCE_JSONL="${INPUT_JSONL:-${REPO_ROOT}/all_selected_merged_current_paths.jsonl}"
SPLIT_ROOT="${SPLIT_ROOT:-${REPO_ROOT}/output/domain_inputs/full_10domain}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/output/domain_full_runs}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"

# Rollout/vLLM 使用的 GPU。
# 支持通过环境变量覆盖，例如：
# ROLLOUT_GPUS=6,7 bash launch_full_domain_train.sh
ROLLOUT_GPUS="${ROLLOUT_GPUS:-6,7}"

# 每张 rollout GPU 至少需要的空闲显存，单位 MB。
ROLLOUT_MIN_FREE_MB="${ROLLOUT_MIN_FREE_MB:-20000}"

# 清理完成后等待显存释放的最长时间。
ROLLOUT_CLEANUP_TIMEOUT="${ROLLOUT_CLEANUP_TIMEOUT:-180}"

# SIGTERM 后等待进程正常退出的时间。
ROLLOUT_TERM_TIMEOUT="${ROLLOUT_TERM_TIMEOUT:-20}"

# 是否允许清理其他用户的进程：
# 0：只清理当前用户的进程；
# 1：也清理其他用户的进程，需要相应权限。
ROLLOUT_KILL_OTHER_USERS="${ROLLOUT_KILL_OTHER_USERS:-0}"

IFS=',' read -r -a ROLLOUT_GPU_ARRAY <<< "${ROLLOUT_GPUS}"

domains=(
  street
  remote
  ground
  map
  indoor
  landmark
  roadnet
  shape
  space
  uav
)

START_DOMAIN="${START_DOMAIN:-}"

if [[ -n "${START_DOMAIN}" ]]; then
  filtered_domains=()
  start_seen=0

  for domain in "${domains[@]}"; do
    if [[ "${domain}" == "${START_DOMAIN}" ]]; then
      start_seen=1
    fi

    if [[ "${start_seen}" -eq 1 ]]; then
      filtered_domains+=("${domain}")
    fi
  done

  if [[ "${#filtered_domains[@]}" -eq 0 ]]; then
    echo "Unknown START_DOMAIN=${START_DOMAIN}. Valid domains: ${domains[*]}" >&2
    exit 2
  fi

  domains=("${filtered_domains[@]}")
fi

mkdir -p "${SPLIT_ROOT}" "${RUN_ROOT}"

###############################################################################
# Rollout GPU 清理函数
###############################################################################

# 判断某个 PID 是否是当前脚本或当前脚本的祖先进程。
# 防止清理时误杀脚本自身、SSH shell、任务调度 shell 等。
is_shell_ancestor() {
  local target_pid="$1"
  local current_pid="$$"
  local parent_pid

  while [[ "${current_pid}" =~ ^[0-9]+$ ]] && (( current_pid > 1 )); do
    if [[ "${current_pid}" == "${target_pid}" ]]; then
      return 0
    fi

    parent_pid="$(
      ps -o ppid= -p "${current_pid}" 2>/dev/null |
        tr -d '[:space:]'
    )"

    if [[ -z "${parent_pid}" ]]; then
      break
    fi

    current_pid="${parent_pid}"
  done

  return 1
}

# 收集占用 rollout GPU 的 PID。
#
# 同时使用两种方式：
# 1. nvidia-smi：获取 CUDA compute process；
# 2. fuser：获取打开 /dev/nvidiaX 的进程。
#
# 有些 vLLM worker 在 nvidia-smi 进程表中可能不可见，
# 因此额外使用 fuser。
collect_rollout_gpu_pids() {
  local gpu

  {
    for gpu in "${ROLLOUT_GPU_ARRAY[@]}"; do
      gpu="${gpu//[[:space:]]/}"

      if [[ -z "${gpu}" ]]; then
        continue
      fi

      nvidia-smi \
        -i "${gpu}" \
        --query-compute-apps=pid \
        --format=csv,noheader,nounits \
        2>/dev/null || true

      if command -v fuser >/dev/null 2>&1; then
        fuser "/dev/nvidia${gpu}" 2>/dev/null || true
      fi
    done
  } |
    tr '[:space:]' '\n' |
    awk '/^[0-9]+$/ { print $1 }' |
    sort -n -u
}

# 获取某张 GPU 的空闲显存，单位 MB。
get_gpu_free_memory_mb() {
  local gpu="$1"
  local free_mb

  free_mb="$(
    nvidia-smi \
      -i "${gpu}" \
      --query-gpu=memory.free \
      --format=csv,noheader,nounits \
      2>/dev/null |
      head -n 1 |
      tr -dc '0-9'
  )"

  if [[ -z "${free_mb}" ]]; then
    return 1
  fi

  printf '%s\n' "${free_mb}"
}

# 判断全部 rollout GPU 是否都有足够空闲显存。
rollout_memory_is_ready() {
  local gpu
  local free_mb

  for gpu in "${ROLLOUT_GPU_ARRAY[@]}"; do
    gpu="${gpu//[[:space:]]/}"

    if [[ -z "${gpu}" ]]; then
      continue
    fi

    free_mb="$(get_gpu_free_memory_mb "${gpu}")" || return 1

    if (( free_mb < ROLLOUT_MIN_FREE_MB )); then
      return 1
    fi
  done

  return 0
}

# 打印 rollout GPU 的当前状态。
print_rollout_gpu_status() {
  local gpu
  local free_mb

  for gpu in "${ROLLOUT_GPU_ARRAY[@]}"; do
    gpu="${gpu//[[:space:]]/}"

    if [[ -z "${gpu}" ]]; then
      continue
    fi

    free_mb="$(get_gpu_free_memory_mb "${gpu}" 2>/dev/null || true)"

    if [[ -n "${free_mb}" ]]; then
      echo "Rollout GPU ${gpu}: free=${free_mb}MB, required>=${ROLLOUT_MIN_FREE_MB}MB"
    else
      echo "Rollout GPU ${gpu}: unable to query free memory" >&2
    fi
  done
}

# 终止当前可见的 rollout GPU 占用进程。
terminate_rollout_gpu_processes() {
  local current_uid
  local pid
  local pid_uid
  local command_line
  local -a pids=()
  local -a killable_pids=()
  local -a remaining_pids=()
  local elapsed

  current_uid="$(id -u)"

  mapfile -t pids < <(collect_rollout_gpu_pids)

  if [[ "${#pids[@]}" -eq 0 ]]; then
    echo "No visible process is using rollout GPUs ${ROLLOUT_GPUS}."
    return 0
  fi

  for pid in "${pids[@]}"; do
    if [[ ! "${pid}" =~ ^[0-9]+$ ]]; then
      continue
    fi

    if [[ ! -d "/proc/${pid}" ]]; then
      continue
    fi

    # 不允许杀掉当前脚本及其祖先进程。
    if is_shell_ancestor "${pid}"; then
      echo "Skipping shell ancestor PID ${pid}."
      continue
    fi

    pid_uid="$(stat -c '%u' "/proc/${pid}" 2>/dev/null || true)"

    if [[ -z "${pid_uid}" ]]; then
      continue
    fi

    if [[ "${ROLLOUT_KILL_OTHER_USERS}" != "1" ]] &&
       [[ "${pid_uid}" != "${current_uid}" ]]; then
      echo "Skipping PID ${pid}: owned by UID ${pid_uid}, current UID=${current_uid}."
      continue
    fi

    command_line="$(
      tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true
    )"

    if [[ -z "${command_line}" ]]; then
      command_line="$(
        ps -o comm= -p "${pid}" 2>/dev/null || true
      )"
    fi

    echo "Found rollout GPU process:"
    echo "  PID=${pid}"
    echo "  UID=${pid_uid}"
    echo "  CMD=${command_line:-unknown}"

    killable_pids+=("${pid}")
  done

  if [[ "${#killable_pids[@]}" -eq 0 ]]; then
    echo "No killable rollout process was found."
    return 0
  fi

  echo "Sending SIGTERM to rollout processes: ${killable_pids[*]}"

  for pid in "${killable_pids[@]}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done

  # 等待进程正常退出。
  for ((elapsed = 0; elapsed < ROLLOUT_TERM_TIMEOUT; elapsed++)); do
    remaining_pids=()

    for pid in "${killable_pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        remaining_pids+=("${pid}")
      fi
    done

    if [[ "${#remaining_pids[@]}" -eq 0 ]]; then
      echo "All rollout processes exited after SIGTERM."
      return 0
    fi

    killable_pids=("${remaining_pids[@]}")
    sleep 1
  done

  # 正常退出超时，使用 SIGKILL。
  echo "SIGTERM timeout; sending SIGKILL to: ${killable_pids[*]}" >&2

  for pid in "${killable_pids[@]}"; do
    kill -KILL "${pid}" 2>/dev/null || true
  done

  sleep 2
}

# 等待 GPU 驱动真正回收显存。
wait_for_rollout_memory() {
  local elapsed=0

  while (( elapsed <= ROLLOUT_CLEANUP_TIMEOUT )); do
    if rollout_memory_is_ready; then
      echo "Rollout GPU memory is ready."
      print_rollout_gpu_status
      return 0
    fi

    if (( elapsed % 15 == 0 )); then
      echo "Waiting for rollout GPU memory to be released..."
      print_rollout_gpu_status
    fi

    sleep 5
    elapsed=$((elapsed + 5))
  done

  echo "ERROR: rollout GPUs did not release enough memory within ${ROLLOUT_CLEANUP_TIMEOUT}s." >&2
  print_rollout_gpu_status >&2

  echo "Visible rollout GPU PIDs:" >&2
  collect_rollout_gpu_pids >&2 || true

  nvidia-smi -i "${ROLLOUT_GPUS}" >&2 || true

  return 1
}

# 完整清理过程。
cleanup_rollout_gpus() {
  echo "============================================================"
  echo "Cleaning rollout GPUs: ${ROLLOUT_GPUS}"
  echo "============================================================"

  terminate_rollout_gpu_processes
  wait_for_rollout_memory
}

# 收到中断信号时清理 GPU。
handle_signal() {
  local signal_name="$1"

  trap - INT TERM

  echo >&2
  echo "Received ${signal_name}; cleaning rollout GPU processes..." >&2

  set +e
  cleanup_rollout_gpus
  set -e

  if [[ "${signal_name}" == "INT" ]]; then
    exit 130
  fi

  exit 143
}

trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM

###############################################################################
# 拆分各 domain 的数据
###############################################################################

python - "${SOURCE_JSONL}" "${SPLIT_ROOT}" "${domains[@]}" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
domains = sys.argv[3:]

handles = {
    domain: (out_dir / f"{domain}.jsonl").open("w", encoding="utf-8")
    for domain in domains
}
counts = {domain: 0 for domain in domains}

try:
    with source.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            domain = row.get("category") or row.get("domain")

            if domain in handles:
                handles[domain].write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )
                counts[domain] += 1
finally:
    for handle in handles.values():
        handle.close()

missing = [
    domain
    for domain, count in counts.items()
    if count == 0
]

if missing:
    raise SystemExit(
        f"Missing domains in {source}: {', '.join(missing)}"
    )

for domain in domains:
    print(
        f"{domain}\t"
        f"{counts[domain]}\t"
        f"{out_dir / (domain + '.jsonl')}"
    )
PY

###############################################################################
# 单个 domain 训练
###############################################################################

run_domain() {
  local domain="$1"
  local run_name
  local log_path
  local status
  local cleanup_status=0

  run_name="geogrpo_qwen35_9b_6p2_b8_full_${domain}_four_stage_${RUN_STAMP}"
  log_path="${RUN_ROOT}/${run_name}.log"

  echo
  echo "################################################################"
  echo "Starting domain: ${domain}"
  echo "Run name: ${run_name}"
  echo "Log: ${log_path}"
  echo "################################################################"

  # 启动当前 domain 前先清理上一个任务可能遗留的 rollout 服务。
  if ! cleanup_rollout_gpus; then
    echo "ERROR: unable to prepare rollout GPUs before ${domain}." >&2
    return 70
  fi

  set +e

  RUN_NAME="${run_name}" \
  INPUT_JSONL="${SPLIT_ROOT}/${domain}.jsonl" \
  TRAIN_DATA_RATIO=1.0 \
  MAX_SAMPLES=all \
  SCHEDULE=four_stage \
  DOMAIN_BALANCE=ratio \
  DOMAIN_RATIO_TEXT="${domain}:1" \
  DOMAIN_ORDER_TEXT="${domain}" \
  SAVE_STRATEGY=epoch \
  SAVE_TOTAL_LIMIT=1 \
  bash "${LAUNCHER}" >"${log_path}" 2>&1

  status=$?

  # 无论 launcher 成功还是失败，都立即清理 rollout 进程。
  cleanup_rollout_gpus
  cleanup_status=$?

  set -e

  # 训练成功但清理失败时，也不能继续下一个 domain。
  if [[ "${cleanup_status}" -ne 0 ]]; then
    echo "ERROR: ${domain} finished, but rollout GPU cleanup failed." >&2

    if [[ "${status}" -eq 0 ]]; then
      status="${cleanup_status}"
    fi
  fi

  if [[ "${status}" -ne 0 ]]; then
    if grep -q \
      "Training process exited with status 0" \
      "${log_path}" 2>/dev/null; then

      echo \
        "WARNING: ${domain} launcher exited with status ${status} after training success; continuing." |
        tee -a "${log_path}" >&2

      return 0
    fi

    echo "ERROR: ${domain} failed with status ${status}; see ${log_path}" >&2
    return "${status}"
  fi

  echo "Domain ${domain} completed successfully."
}

###############################################################################
# 串行训练全部 domain
###############################################################################

for domain in "${domains[@]}"; do
  run_domain "${domain}"
done

echo
echo "All requested domains completed successfully."
