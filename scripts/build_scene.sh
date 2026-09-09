#!/usr/bin/env bash
# One image (plus your tape measure) -> a runnable MuJoCo scene.
#
#   scripts/build_scene.sh --image inputs/frame0.png --out outputs/kitchen \
#       --objects "the wooden table,the red mug" \
#       --measure obj0_wooden_table=0.745
#
# Chains the whole object half of the pipeline: segmentation, meshes, collision
# hulls, physical properties, MJCF export and metric rescale. Add a robot with
# --robot to finish, then run it with scripts/realsim/start.sh.
#
# Roughly 20 minutes and ~17 GB of VRAM. Every stage is skippable on re-run, so
# an interrupted build resumes rather than starting over.
#
# The background splat is NOT built here - it needs a video and its own ~15
# minutes. See README section 4.4.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/_bootstrap.sh"
cd "$ROOT"

IMAGE=""; OUT=""; OBJECTS=""; MEASURE=""; ROBOT=""; AT=""; YAW="0"
STATIC="table"; CHECK=""
usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --image)   IMAGE="$2"; shift 2 ;;
    --out)     OUT="$2"; shift 2 ;;
    --objects) OBJECTS="$2"; shift 2 ;;
    --measure) MEASURE="$2"; shift 2 ;;   # BODY=METRES, the real dimension
    --check)   CHECK="$2"; shift 2 ;;     # BODY=W,H  a second measurement
    --robot)   ROBOT="$2"; shift 2 ;;
    --at)      AT="$2 $3"; shift 3 ;;
    --yaw)     YAW="$2"; shift 2 ;;
    --static)  STATIC="$2"; shift 2 ;;
    -h|--help) usage 0 ;;
    *) echo "unknown flag: $1" >&2; usage 2 ;;
  esac
done

[ -n "$IMAGE" ]   || { echo "--image is required" >&2; usage 2; }
[ -n "$OUT" ]     || { echo "--out is required" >&2; usage 2; }
[ -n "$OBJECTS" ] || { echo "--objects is required" >&2; usage 2; }
if [ ! -f "$IMAGE" ]; then
  echo "no such image: $IMAGE" >&2
  echo "images available:" >&2
  ls -1 inputs/*.png inputs/*.jpg examples/*.png 2>/dev/null | sed 's/^/    /' >&2
  exit 1
fi
if [ -z "$MEASURE" ]; then
  cat >&2 <<'EOF'
--measure is required, and it is the one number you cannot guess.

Reconstruction is scale-free: it recovers correct proportions and an arbitrary
absolute size. One real dimension sets the true size of everything else. Measure
it with a tape.

    --measure obj0_wooden_table=0.745      (body name = height in metres)

Body names come from --objects, in order: obj0_<slug>, obj1_<slug>, ...
EOF
  exit 2
fi
MEAS_BODY="${MEASURE%%=*}"; MEAS_VAL="${MEASURE#*=}"

step() { printf '\n\033[1m== [%d/%d] %s\033[0m\n' "$1" "$2" "$3"; }
TOTAL=4; [ -n "$ROBOT" ] && TOTAL=5

step 1 $TOTAL "objects: masks, meshes, collision hulls  (the long one)"
if [ -f "$OUT/fully_refined_foreground.glb" ]; then
  echo "  already done, skipping"
else
  INPUT_IMAGE="$IMAGE" OUTPUT_DIR="$OUT" OBJECT_PROMPTS="$OBJECTS" \
  SF_REAL2SIM_COLLISION_METHOD="${SF_REAL2SIM_COLLISION_METHOD:-coacd}" \
    bash "$ROOT/scripts/run_object_pipeline.sh"
fi
[ -f "$OUT/fully_refined_foreground.glb" ] || {
  echo; echo "stage 1 did not finish. The mask stage is the usual cause -" >&2
  echo "inspect $OUT/masks/ and refine them in the GUI (bash scripts/run_gui.sh)," >&2
  echo "then re-run this command; finished stages are skipped." >&2
  echo "full log: $OUT/pipeline.log" >&2; exit 1; }

step 2 $TOTAL "physical properties"
"$RS_PY_SIM" scripts/estimate_properties.py --scene-dir "$OUT" --mode lookup

step 3 $TOTAL "MuJoCo export"
MUJOCO_GL=egl "$RS_PY_SIM" scripts/export_mujoco.py \
    --scene-dir "$OUT" --static "$STATIC"

step 4 $TOTAL "metric rescale, anchored on $MEAS_BODY = ${MEAS_VAL} m"
RESCALE=(--scene "$OUT/mujoco/scene.xml"
         --output "$OUT/mujoco/scene_scaled.xml"
         --table-body "$MEAS_BODY" --table-height "$MEAS_VAL")
if [ -n "$CHECK" ]; then
  CB="${CHECK%%=*}"; CV="${CHECK#*=}"
  RESCALE+=(--basket-body "$CB" --check-basket "${CV%%,*}" "${CV##*,}")
fi
MUJOCO_GL=egl "$RS_PY_SIM" scripts/rescale_scene.py "${RESCALE[@]}"

FINAL="$OUT/mujoco/scene_scaled.xml"
if [ -n "$ROBOT" ]; then
  step 5 $TOTAL "place the robot"
  if [ -z "$AT" ]; then
    echo "  --robot needs --at X Y (where it stands, scene coords)" >&2
    echo "  open the scene to pick a spot:" >&2
    echo "    MUJOCO_GL=glfw $RS_PY_SIM -m mujoco.viewer --mjcf=$FINAL" >&2
    exit 2
  fi
  RMJCF=$(ls "assets/robots/$ROBOT"/*.xml 2>/dev/null | head -1)
  [ -n "$RMJCF" ] || { echo "  no MJCF in assets/robots/$ROBOT" >&2
                       echo "  fetch one: scripts/fetch_robot.sh --list" >&2; exit 1; }
  MUJOCO_GL=egl "$RS_PY_SIM" scripts/add_robot.py \
      --scene-dir "$OUT" --scene-xml "$FINAL" --robot "$RMJCF" \
      --at $AT --yaw "$YAW" --strip-rangefinders \
      --output "$OUT/mujoco/scene_robot.xml"
  FINAL="$OUT/mujoco/scene_robot.xml"
fi

cat <<EOF

$(printf '\033[1mscene ready:\033[0m') $FINAL

Look at it:
    MUJOCO_GL=glfw $RS_PY_SIM -m mujoco.viewer --mjcf=$FINAL

Run it (sim + hybrid + robot cameras):
    scripts/realsim/start.sh --scene $OUT

The hybrid render needs a background splat, which this script does not build -
it wants a video of the room. See README section 4.4. Until then the sim works
and the render shows a plain floor.
EOF
