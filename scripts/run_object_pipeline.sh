#!/usr/bin/env bash
# Object-asset pipeline: image -> simulation-ready object meshes.
#
# Runs the same stages as scripts/run_auto_pipeline.sh but stops short of the
# Lyra-2 background layer, which needs ~48GB VRAM. Everything the objects
# depend on (masks, SAM3D meshes, VGGT camera, ICP, gravity alignment,
# SF-Real2Sim physics refinement) runs without it.
#
#   INPUT_IMAGE=/path/img.png OUTPUT_DIR=/path/out bash scripts/run_object_pipeline.sh
set -uo pipefail

ROOT="${ROBOSNAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
set -a; . "$ROOT/configs/auto_pipeline.env"; set +a
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

INPUT_IMAGE="${INPUT_IMAGE:-$ROOT/examples/test1.png}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/objects}"
# VHACD can consume >60GB RAM and get OOM-killed; CoACD is orders of magnitude lighter.
export SF_REAL2SIM_COLLISION_METHOD="${SF_REAL2SIM_COLLISION_METHOD:-coacd}"

step() { printf '\n=== [object-pipeline] %s ===\n' "$*"; }

# Check the input before spending 20 minutes discovering it is missing. Without
# this the failure surfaces three stack traces deep, as a FileNotFoundError
# inside auto_segment_image, and the two follow-up stages then fail for their
# own reasons on top of it.
if [ ! -f "$INPUT_IMAGE" ]; then
  echo "[object-pipeline] INPUT_IMAGE does not exist: $INPUT_IMAGE" >&2
  echo "[object-pipeline] images available under $ROOT/inputs:" >&2
  ls -1 "$ROOT/inputs" 2>/dev/null | sed 's/^/    /' >&2 || echo "    (no inputs/ directory)" >&2
  exit 2
fi

step "stages 1-4: preprocess, sam3d, vggt, icp  (background stage will stop it; expected)"
INPUT_IMAGE="$INPUT_IMAGE" OUTPUT_DIR="$OUTPUT_DIR" SKIP_EXISTING=1 \
  bash "$ROOT/scripts/run_auto_pipeline.sh" || true

if [ ! -f "$OUTPUT_DIR/refined_scene.glb" ]; then
  echo "[object-pipeline] FAILED before ICP: no refined_scene.glb. See $OUTPUT_DIR/pipeline.log" >&2
  exit 1
fi

step "stage 5: gravity alignment (no background ply - it is optional)"
"$PY_ALIGN" -m robosnap.refinement.gravity_align_scene \
  --scene-dir "$OUTPUT_DIR" \
  --foreground-glb "$OUTPUT_DIR/refined_scene.glb" \
  --vggt-dir "$OUTPUT_DIR/depth" \
  --plane-mask "$OUTPUT_DIR/support_mask.png" \
  --output-foreground "$OUTPUT_DIR/gravity_aligned_foreground.glb" \
  --output-background "$OUTPUT_DIR/gravity_aligned_background.ply" \
  --transform-json "$OUTPUT_DIR/gravity_alignment.json" \
  --seed "${SEED:-0}" || { echo "[object-pipeline] gravity alignment failed" >&2; exit 1; }

step "stage 6: SF-Real2Sim physics refinement (collision meshes + pose settling)"
"$PY_ASSET" -m robosnap.refinement.run_sim_ready_refinement \
  --input-foreground "$OUTPUT_DIR/gravity_aligned_foreground.glb" \
  --output-foreground "$OUTPUT_DIR/fully_refined_foreground.glb" \
  --refinement-dir "$OUTPUT_DIR/refinement" \
  --scene-dir "$OUTPUT_DIR" \
  --sf-python "$PY_SIM_READY" || { echo "[object-pipeline] refinement failed" >&2; exit 1; }

step "done"
ls -la "$OUTPUT_DIR/fully_refined_foreground.glb"
