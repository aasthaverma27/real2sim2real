#!/usr/bin/env bash
# Put a robot model into assets/robots/, from MuJoCo Menagerie or a local file.
#
#   scripts/fetch_robot.sh --list
#   scripts/fetch_robot.sh unitree_g1
#   scripts/fetch_robot.sh --from /path/to/robot.xml --name my_robot
#
# Models are NOT committed to this repository - they belong to their
# manufacturers, and assets/robots/*/meshes/ is gitignored. This script only
# puts one on your disk.
#
# Fetching a model does not make it drivable by a whole-body controller. See
# assets/robots/README.md for the invariants a controller assumes, and describe
# the model in configs/robots/<name>.yaml before using it.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DEST="$ROOT/assets/robots"
MENAGERIE="https://github.com/google-deepmind/mujoco_menagerie.git"

# Menagerie directory names for humanoids and arms people actually use here.
KNOWN="unitree_g1 unitree_h1 unitree_z1 booster_t1 berkeley_humanoid
       pal_talos pal_tiago robotis_op3 agility_cassie
       franka_emika_panda franka_fr3 universal_robots_ur5e
       kuka_iiwa_14 rethink_robotics_sawyer trs_so_arm100
       shadow_hand wonik_allegro leap_hand"

usage() { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; }

if [ $# -eq 0 ]; then usage; exit 2; fi

case "$1" in
  --list)
    echo "MuJoCo Menagerie models this script knows about:"
    for k in $KNOWN; do echo "    $k"; done
    echo
    echo "Any other Menagerie directory name works too - pass it directly."
    echo "Full catalogue: https://github.com/google-deepmind/mujoco_menagerie"
    exit 0 ;;
  --from)
    SRC="${2:?--from needs a path to an MJCF}"
    NAME=""
    [ "${3:-}" = "--name" ] && NAME="${4:?--name needs a value}"
    [ -f "$SRC" ] || { echo "no such file: $SRC" >&2; exit 1; }
    NAME="${NAME:-$(basename "${SRC%.*}")}"
    OUT="$DEST/$NAME"
    mkdir -p "$OUT"
    cp "$SRC" "$OUT/"
    # Bring the meshes the MJCF actually references, wherever they live.
    MESHDIR=$(grep -oE 'meshdir="[^"]*"' "$SRC" | head -1 | cut -d'"' -f2 || true)
    if [ -n "$MESHDIR" ]; then
      ABS="$(cd "$(dirname "$SRC")" && cd "$MESHDIR" 2>/dev/null && pwd || true)"
      if [ -n "$ABS" ]; then
        mkdir -p "$OUT/$MESHDIR"; cp -r "$ABS/." "$OUT/$MESHDIR/"
        echo "[fetch] copied meshes from $ABS"
      else
        echo "[fetch] WARNING: MJCF wants meshdir='$MESHDIR' but it was not found"
        echo "[fetch] next to $SRC - copy it into $OUT/$MESHDIR yourself."
      fi
    fi
    echo "[fetch] $NAME -> $OUT"
    ;;
  -h|--help) usage; exit 0 ;;
  *)
    NAME="$1"
    OUT="$DEST/$NAME"
    if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
      echo "[fetch] $NAME already present at $OUT"; exit 0
    fi
    TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
    echo "[fetch] cloning Menagerie (sparse, shallow) for '$NAME'"
    git clone --depth 1 --filter=blob:none --sparse --quiet "$MENAGERIE" "$TMP/m"
    git -C "$TMP/m" sparse-checkout set "$NAME" >/dev/null 2>&1 || true
    if [ ! -d "$TMP/m/$NAME" ]; then
      echo "[fetch] Menagerie has no model directory '$NAME'." >&2
      echo "[fetch] try: scripts/fetch_robot.sh --list" >&2
      exit 1
    fi
    mkdir -p "$OUT"; cp -r "$TMP/m/$NAME/." "$OUT/"
    echo "[fetch] $NAME -> $OUT"
    [ -f "$OUT/LICENSE" ] && echo "[fetch] its licence: $OUT/LICENSE"
    ;;
esac

cat <<EOF

Next:
  1. describe it     cp configs/robots/g1_29dof_with_hand.yaml configs/robots/$NAME.yaml
                     \$EDITOR configs/robots/$NAME.yaml
  2. check the model meets a controller's invariants (assets/robots/README.md)
  3. place it        scripts/add_robot.py --robot assets/robots/$NAME/<file>.xml ...
EOF
