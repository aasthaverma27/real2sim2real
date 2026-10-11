#!/usr/bin/env bash
# Wait for the shared card to free up, then run a command.
#
# The 5090 is shared and the working ceiling is 31 GB. A heavy stage launched
# against a full card does not merely fail - it can OOM whoever else is on it,
# which is why build_sim_scene.sh refuses rather than tries. This is the other
# half of that: park until there is room, then go.
#
#     bash scripts/wait_for_vram.sh 8000 -- bash scripts/build_sim_scene.sh ...
#
# First argument is the headroom needed in MiB. Polls every 2 minutes and gives
# up after 6 hours rather than sitting forever.
set -uo pipefail
NEED="${1:?need headroom MiB}"; shift
[ "${1:-}" = "--" ] && shift
TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
DEADLINE=$(( $(date +%s) + 6*3600 ))
while :; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  FREE=$(( TOTAL - USED ))
  if [ "$FREE" -ge "$NEED" ]; then
    echo "[wait-vram] $(date +%H:%M:%S) ${FREE} MiB free >= ${NEED} MiB - running"
    exec "$@"
  fi
  [ "$(date +%s)" -ge "$DEADLINE" ] && {
    echo "[wait-vram] gave up after 6h; ${FREE} MiB free, needed ${NEED}" >&2; exit 1; }
  echo "[wait-vram] $(date +%H:%M:%S) ${FREE} MiB free, need ${NEED} - waiting"
  sleep 120
done
