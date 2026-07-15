#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="${1:?usage: a4_repair_probe.sh LOG_FILE}"
mkdir -p "$(dirname "${LOG_FILE}")"

{
  echo "START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "HOST=$(hostname)"
  echo "PID=$$"
  echo "A4_REPAIR_PROBE_RUNNING"
  echo "SNAPSHOT_BEFORE_LOOP"
  nvidia-smi || true
  echo "PIDS_BEFORE_LOOP"
  pgrep -af 'cosmos_cot8|cosmos_cot.py|a4_idle|last05_test4|a4_repair_probe' || true
} >>"${LOG_FILE}" 2>&1

cleanup() {
  {
    echo "TERM $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "SNAPSHOT_ON_TERM"
    nvidia-smi || true
    echo "PIDS_ON_TERM"
    pgrep -af 'cosmos_cot8|cosmos_cot.py|a4_idle|last05_test4|a4_repair_probe' || true
  } >>"${LOG_FILE}" 2>&1
  exit 0
}
trap cleanup TERM INT

while true; do
  echo "TICK $(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"${LOG_FILE}"
  sleep 5
done
