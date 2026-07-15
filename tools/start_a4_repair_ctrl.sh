#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/nas/zhangyiming/geobench"
SUPERVISOR="/mnt/nas/zhangyiming/train/bash/last05_test4.sh"
IDLE_SCRIPT="${REPO_ROOT}/tools/a4_idle.sh"
CONTROL_FILE="${REPO_ROOT}/a4_repair.txt"
OLD_CONTROL_FILE="${REPO_ROOT}/a4.txt"
LOG_FILE="${REPO_ROOT}/gpu8_repair.log"
OLD_SUPPRESS_COMMAND="1 bash -lc 'while true; do sleep 3600; done'"
OLD_SUPPRESS_WAIT_SECONDS="${OLD_SUPPRESS_WAIT_SECONDS:-30}"

cat > "${IDLE_SCRIPT}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

# Default background job for the repaired A4 controller. This intentionally
# keeps the 8-card machine busy when no training override is active.
exec env \
  PHYSICAL_GPU_IDS="${PHYSICAL_GPU_IDS:-0,1,2,3,4,5,6,7}" \
  WANDB_MODE="${WANDB_MODE:-offline}" \
  bash /mnt/nas/zhangyiming/train/bash/cosmos_cot8.sh
EOF
chmod +x "${IDLE_SCRIPT}"

tmux kill-session -t a4ctrl 2>/dev/null || true
printf '%s\n' "${OLD_SUPPRESS_COMMAND}" > "${OLD_CONTROL_FILE}"
echo "Suppressing old a4 controller for ${OLD_SUPPRESS_WAIT_SECONDS}s: ${OLD_CONTROL_FILE}"
sleep "${OLD_SUPPRESS_WAIT_SECONDS}"
printf '0\n' > "${CONTROL_FILE}"
: > "${LOG_FILE}"

tmux new-session -d -s a4ctrl \
  "exec env DEFAULT_SCRIPT='${IDLE_SCRIPT}' CONTROL_FILE='${CONTROL_FILE}' CHECK_INTERVAL=10 bash '${SUPERVISOR}' >> '${LOG_FILE}' 2>&1"

sleep 2
tmux ls
tail -n 80 "${LOG_FILE}"
