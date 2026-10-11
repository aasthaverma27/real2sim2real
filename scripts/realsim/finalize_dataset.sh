#!/usr/bin/env bash
# Turn recording sessions into one clean LeRobot dataset.
#
# The exporter's 'x' does NOT delete an episode: it saves it and lists its index
# under discarded_episode_indices in meta/info.json. This drops those, renumbers
# what is left, and merges several sessions into one dataset, using
# GR00T-WholeBodyControl/gear_sonic/scripts/process_dataset.py - the tool
# that already speaks this exporter's format. It runs in the WBC container, the
# only environment here with av + pandas + pyarrow + tyro.
#
# Sources are only touched to fold operator_discarded.json into their
# info.json discard list; the output directory must not exist yet.
#
#   scripts/realsim/finalize_dataset.sh <output-name> <session> [<session> ...]
#   scripts/realsim/finalize_dataset.sh apple-pnp-G1-sim \
#       2026-09-29-11-20-00-G1-sim 2026-09-29-12-05-31-G1-sim
#
# Names are folders under $RS_DEPLOY_ROOT/outputs/, which is where the
# exporter (running in the WBC container) writes them.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
. "$ROOT/scripts/_bootstrap.sh" || exit 1
HOST_OUT="$RS_DEPLOY_ROOT/outputs"
CT_OUT=/root/Projects/GR00T-WholeBodyControl/outputs
TOOL="$RS_DEPLOY_ROOT/gear_sonic/scripts/process_dataset.py"

[ $# -ge 2 ] || { sed -n 2,18p "$0"; exit 2; }
NAME="$1"; shift
[ -e "$HOST_OUT/$NAME" ] && { echo "$HOST_OUT/$NAME already exists" >&2; exit 1; }

SRC=()
for s in "$@"; do
  s="${s%/}"; s="${s##*/}"
  [ -f "$HOST_OUT/$s/meta/info.json" ] || { echo "not a dataset: $HOST_OUT/$s" >&2; exit 1; }
  python3 - "$HOST_OUT/$s" <<'PY'
import json, os, sys
i = json.load(open(sys.argv[1] + "/meta/info.json"))
d = set(i.get("discarded_episode_indices", []))
side = sys.argv[1] + "/meta/operator_discarded.json"
if os.path.isfile(side):
    d |= set(json.load(open(side)))
print(f"  {sys.argv[1].rsplit('/', 1)[-1]}: {i['total_episodes']} episodes, "
      f"{len(d)} flagged discarded -> {i['total_episodes'] - len(d)} kept")
PY
  SRC+=("$CT_OUT/$s")
done

CONTAINER="${WBC_CONTAINER:-$(docker ps --format '{{.Names}}' | grep -m1 '^decoupled_wbc-' || true)}"
[ -n "$CONTAINER" ] || { echo "no decoupled_wbc container running" >&2; exit 1; }
docker exec "$CONTAINER" mkdir -p /root/collect
docker cp "$TOOL" "$CONTAINER:/root/collect/process_dataset.py" >/dev/null

# Discards decided AFTER an episode was saved live in
# meta/operator_discarded.json, because the exporter rewrites info.json on every
# save. Fold them in now that the session is finished. This is the only change
# ever made to a source dataset.
docker exec -i "$CONTAINER" python3 - "${SRC[@]}" <<'PY'
import json, os, sys
for d in sys.argv[1:]:
    side = os.path.join(d, "meta", "operator_discarded.json")
    if not os.path.isfile(side):
        continue
    p = os.path.join(d, "meta", "info.json")
    info = json.load(open(p))
    merged = sorted(set(info.get("discarded_episode_indices", []))
                    | set(json.load(open(side))))
    if merged != info.get("discarded_episode_indices", []):
        info["discarded_episode_indices"] = merged
        json.dump(info, open(p, "w"), indent=4)
        print(f"[finalize] {os.path.basename(d)}: discarded {merged}")
PY

# --no-remove-stale-smpl: that pass is for SMPL teleop streams, which Pico
# teleop in this sim does not produce.
docker exec "$CONTAINER" bash -c '
  source /root/venv/bin/activate
  python /root/collect/process_dataset.py --dataset-path "${@:2}" \
      --output-path "$1" --no-remove-stale-smpl' _ "$CT_OUT/$NAME" "${SRC[@]}" \
  | grep -vE "^\s*Episode [0-9]+: (kept|discarded)" || true

[ -f "$HOST_OUT/$NAME/meta/info.json" ] || { echo "no output written" >&2; exit 1; }
python3 - "$HOST_OUT/$NAME" <<'PY'
import json, sys
a = json.load(open(sys.argv[1] + "/meta/info.json"))
print(f"[finalize] {sys.argv[1]}: {a['total_episodes']} episodes, "
      f"{a['total_frames']} frames, LeRobot {a['codebase_version']} @ {a['fps']} fps")
PY
