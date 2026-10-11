#!/usr/bin/env bash
# Check that this machine can run the pipeline, and say what to do about
# anything that is missing. Changes nothing.
#
#   scripts/doctor.sh
#
# Ends with "ready" when the scene pipeline can run. Teleop / dataset recording
# additionally needs the rows marked [teleop].
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FAIL=0
ok()   { printf '  [ ok ] %s\n' "$1"; }
bad()  { printf '  [FAIL] %s\n         -> %s\n' "$1" "$2"; FAIL=1; }
warn() { printf '  [ -- ] %s\n         -> %s\n' "$1" "$2"; }

echo "config"
if [ -f configs/paths.env ]; then
  ok "configs/paths.env"
  # shellcheck disable=SC1091
  . configs/paths.env
else
  bad "configs/paths.env missing" "bash scripts/install_auto_pipeline.sh  (or copy configs/paths.env.example)"
fi

echo "python environments"
check_env() {   # check_env <var> <imports> <fix>
  local py="${!1:-}"
  if [ -z "$py" ] || [ ! -x "$py" ]; then
    bad "$1 not set or not executable (${py:-unset})" "$3"; return
  fi
  if out=$("$py" -c "import $2" 2>&1); then ok "$1  ($2)"
  else bad "$1 cannot import: $2" "$3   [$(echo "$out" | tail -1)]"; fi
}
check_env RS_PY_SIM   "mujoco, gsplat, cv2, zmq, plyfile, scipy, onnxruntime, yaml" \
  "bash scripts/install_auto_pipeline.sh --only sim"
check_env RS_PY_LYRA  "pycolmap, plyfile, gsplat, torch" \
  "bash scripts/install_auto_pipeline.sh --only lyra"
check_env RS_PY_ASSET "sam3d_objects, vggt, trimesh, open3d" \
  "bash scripts/install_auto_pipeline.sh --only asset"
check_env RS_PY_SAM3  "sam3" \
  "bash scripts/install_auto_pipeline.sh --only sam"

echo "source trees and weights"
for d in sam3 sam-3d-objects vggt; do
  if [ -n "$(ls -A "third_party/$d" 2>/dev/null)" ]; then ok "third_party/$d"
  else bad "third_party/$d empty" "bash scripts/setup_auto_sources.sh"; fi
done
[ -d third_party/MuGS/src ] && ok "third_party/MuGS (vendored)" \
  || bad "third_party/MuGS missing" "it is part of this repository - re-clone"
for f in checkpoints/sam3/sam3.pt checkpoints/sam-3d-objects/pipeline.yaml; do
  if [ -e "$f" ]; then ok "$f"
  else bad "$f missing" "bash scripts/install_auto_pipeline.sh --download-checkpoints"; fi
done

echo "robot"
if ls assets/robots/g1/meshes/*.STL >/dev/null 2>&1; then ok "G1 meshes"
else bad "G1 meshes missing" "bash scripts/fetch_g1_meshes.sh"; fi
[ -f assets/robots/g1/meshes/cad/cad_configuration_1_v3_16.stl ] && ok "camera-rig CAD meshes" \
  || bad "camera-rig CAD meshes missing" "they ship with the repo - re-clone"

echo "system"
for b in ffmpeg nvidia-smi; do
  command -v "$b" >/dev/null && ok "$b" || bad "$b not on PATH" "install it (apt / your driver)"
done
if command -v nvidia-smi >/dev/null; then
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  ok "GPU: ${USED}/${TOTAL} MiB used, ceiling ${RS_VRAM_CEILING_MIB:-31000} MiB"
fi
[ -n "${DISPLAY:-}" ] && ok "DISPLAY=$DISPLAY (windows)" \
  || warn "DISPLAY unset" "the viewer windows need a display; set DISPLAY in configs/paths.env"
command -v opencv_interactive-calibration >/dev/null && ok "opencv_interactive-calibration (calibration)" \
  || warn "opencv_interactive-calibration not found (calibration only)" "sudo apt install libopencv-dev"

echo "[teleop] deployed control stack"
if [ -f "${RS_DEPLOY_ROOT:-/nonexistent}/decoupled_wbc/control/main/teleop/run_g1_control_loop.py" ]; then
  ok "RS_DEPLOY_ROOT=${RS_DEPLOY_ROOT:-unset}"
  if [ -n "${RS_PY_DDS:-}" ] && "$RS_PY_DDS" -c "import unitree_sdk2py, gear_sonic" >/dev/null 2>&1; then
    ok "RS_PY_DDS (unitree_sdk2py, gear_sonic)"
  else
    warn "RS_PY_DDS cannot import unitree_sdk2py / gear_sonic" \
      "bash scripts/setup_external.sh   (builds .venv_sim)"
  fi
  command -v docker >/dev/null && ok "docker" || warn "docker not found" "needed for the WBC container"
else
  warn "GR00T-WholeBodyControl not set up" \
    "bash scripts/setup_external.sh; without it use start.sh --builtin"
fi

echo
if [ "$FAIL" = 0 ]; then echo "ready"; else echo "not ready - fix the [FAIL] rows above"; exit 1; fi
