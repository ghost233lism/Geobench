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

python - "${SOURCE_JSONL}" "${SPLIT_ROOT}" "${domains[@]}" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
domains = sys.argv[3:]

handles = {domain: (out_dir / f"{domain}.jsonl").open("w", encoding="utf-8") for domain in domains}
counts = {domain: 0 for domain in domains}

try:
    with source.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            domain = row.get("category") or row.get("domain")
            if domain in handles:
                handles[domain].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[domain] += 1
finally:
    for handle in handles.values():
        handle.close()

missing = [domain for domain, count in counts.items() if count == 0]
if missing:
    raise SystemExit(f"Missing domains in {source}: {', '.join(missing)}")

for domain in domains:
    print(f"{domain}\t{counts[domain]}\t{out_dir / (domain + '.jsonl')}")
PY

run_domain() {
  local domain="$1"
  local run_name="geogrpo_qwen35_9b_6p2_b8_full_${domain}_four_stage_${RUN_STAMP}"
  local log_path="${RUN_ROOT}/${run_name}.log"
  local status

  echo "=== ${domain}: ${run_name} ==="
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
  bash "${LAUNCHER}" > "${log_path}" 2>&1
  status=$?
  set -e

  if [[ "${status}" -ne 0 ]]; then
    if grep -q "Training process exited with status 0" "${log_path}" 2>/dev/null; then
      echo "WARNING: ${domain} launcher exited with status ${status} after training success; continuing." | tee -a "${log_path}" >&2
      return 0
    fi
    echo "ERROR: ${domain} failed with status ${status}; see ${log_path}" >&2
    return "${status}"
  fi
}

for domain in "${domains[@]}"; do
  run_domain "${domain}"
done
