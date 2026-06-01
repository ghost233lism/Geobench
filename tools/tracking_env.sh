#!/usr/bin/env bash
# Shared tracking setup for GeoBench training entrypoints.

geobench_source_tracking_keys() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  local conf
  for conf in "${script_dir}/wandb_api_key.conf" "${script_dir}/tracking_keys.conf"; do
    if [[ -f "${conf}" ]]; then
      # shellcheck source=/dev/null
      source "${conf}"
    fi
  done
}

geobench_configure_tracking() {
  local run_name="${1:-${RUN_NAME:-geobench}}"
  geobench_source_tracking_keys

  export WANDB_PROJECT="${WANDB_PROJECT:-Geobench}"
  export WANDB_NAME="${WANDB_NAME:-${run_name}}"
  export WANDB_MODE="${WANDB_MODE:-online}"

  local report_targets=()
  if [[ -n "${WANDB_API_KEY:-}" && "${WANDB_DISABLED:-false}" != "true" ]]; then
    export WANDB_DISABLED="${WANDB_DISABLED:-false}"
    report_targets+=("wandb")
  else
    export WANDB_DISABLED="true"
  fi

  if [[ -n "${SWANLAB_API_KEY:-}" ]]; then
    export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
    export SWANLAB_PROJECT="${SWANLAB_PROJECT:-GeoBench-RL}"
    export SWANLAB_EXP_NAME="${SWANLAB_EXP_NAME:-${run_name}}"
    report_targets+=("swanlab")
  else
    export SWANLAB_MODE="${SWANLAB_MODE:-disabled}"
  fi

  if [[ -z "${REPORT_TO:-}" ]]; then
    if (( ${#report_targets[@]} > 0 )); then
      REPORT_TO="${report_targets[*]}"
    else
      REPORT_TO="none"
    fi
  fi
  export REPORT_TO

  if [[ "${REPORT_TO}" == "none" ]]; then
    export DO_NOT_TRACK="${DO_NOT_TRACK:-1}"
  else
    export DO_NOT_TRACK="${DO_NOT_TRACK:-0}"
  fi
}
