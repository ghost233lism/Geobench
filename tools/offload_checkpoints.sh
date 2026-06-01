#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${1:?source output dir required}"
TARGET_ROOT="${2:?target checkpoint root required}"
RUN_NAME="${3:?run name required}"
SAVE_TOTAL_LIMIT="${4:-0}"

INTERVAL_SECONDS="${CHECKPOINT_OFFLOAD_INTERVAL_SECONDS:-30}"
STABLE_SECONDS="${CHECKPOINT_OFFLOAD_STABLE_SECONDS:-20}"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

checkpoint_step() {
  basename "$1" | sed -E 's/^checkpoint-([0-9]+)$/\1/'
}

checkpoint_complete() {
  local ckpt="$1"
  [[ -d "${ckpt}" ]] || return 1
  [[ -f "${ckpt}/trainer_state.json" ]] || return 1
  find "${ckpt}" -maxdepth 2 \( -name '*.safetensors' -o -name '*.bin' -o -name 'latest' \) -print -quit | grep -q .
}

checkpoint_stable() {
  local ckpt="$1"
  local before after
  before="$(du -sb "${ckpt}" 2>/dev/null | awk '{print $1}')" || return 1
  sleep "${STABLE_SECONDS}"
  [[ -d "${ckpt}" ]] || return 1
  after="$(du -sb "${ckpt}" 2>/dev/null | awk '{print $1}')" || return 1
  [[ "${before}" == "${after}" ]]
}

enforce_limit() {
  local version_dir="$1"
  local link_dir="$2"

  [[ "${SAVE_TOTAL_LIMIT}" =~ ^[0-9]+$ ]] || return 0
  (( SAVE_TOTAL_LIMIT > 0 )) || return 0

  mapfile -t checkpoints < <(
    find "${version_dir}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -print 2>/dev/null \
      | sort -t- -k2,2n
  )
  local excess=$(( ${#checkpoints[@]} - SAVE_TOTAL_LIMIT ))
  (( excess > 0 )) || return 0

  local i ckpt step link_path
  for (( i = 0; i < excess; i++ )); do
    ckpt="${checkpoints[$i]}"
    step="$(checkpoint_step "${ckpt}")"
    link_path="${link_dir}/checkpoint-${step}"
    log "removing old offloaded checkpoint ${ckpt}"
    rm -rf "${ckpt}"
    if [[ -L "${link_path}" ]]; then
      rm -f "${link_path}"
    fi
  done
}

offload_checkpoint() {
  local ckpt="$1"
  [[ -L "${ckpt}" ]] && return 0
  checkpoint_complete "${ckpt}" || return 0
  checkpoint_stable "${ckpt}" || return 0

  local version_name ckpt_name dest_dir dest_parent
  version_name="$(basename "$(dirname "${ckpt}")")"
  ckpt_name="$(basename "${ckpt}")"
  dest_dir="${TARGET_ROOT}/${RUN_NAME}/${version_name}/${ckpt_name}"
  dest_parent="$(dirname "${dest_dir}")"

  if [[ -e "${dest_dir}" ]]; then
    log "target already exists, leaving source in place: ${dest_dir}"
    return 0
  fi

  mkdir -p "${dest_parent}" || {
    log "target root unavailable, retrying later: ${TARGET_ROOT}"
    return 0
  }

  log "offloading ${ckpt} -> ${dest_dir}"
  mv "${ckpt}" "${dest_dir}"
  ln -s "${dest_dir}" "${ckpt}"
  enforce_limit "${dest_parent}" "$(dirname "${ckpt}")"
}

scan_checkpoints() {
  if [[ -d "${SOURCE_DIR}" ]]; then
    while IFS= read -r ckpt; do
      offload_checkpoint "${ckpt}"
    done < <(find "${SOURCE_DIR}" -mindepth 2 -maxdepth 2 -type d -name 'checkpoint-*' -print 2>/dev/null | sort)
  fi
}

if [[ "${CHECKPOINT_OFFLOAD_ONCE:-false}" == "true" ]]; then
  log "one-shot checkpoint offload on host $(hostname); source=${SOURCE_DIR}; target=${TARGET_ROOT}/${RUN_NAME}"
  scan_checkpoints
  exit 0
fi

log "checkpoint offloader host: $(hostname)"
log "watching ${SOURCE_DIR}; offloaded checkpoints go to ${TARGET_ROOT}/${RUN_NAME}"

while true; do
  scan_checkpoints
  sleep "${INTERVAL_SECONDS}"
done
