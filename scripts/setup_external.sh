#!/usr/bin/env bash
# Fetch NVIDIA's GR00T-WholeBodyControl (the whole-body controller, Pico
# teleop, LeRobot exporter and DDS sim bridge), apply our changes to it, and
# build the DDS environment the simulator runs in.
#
#   scripts/setup_external.sh            fetch + patch + model files + .venv_sim
#   scripts/setup_external.sh --check    report only, change nothing
#   scripts/setup_external.sh --no-venv  skip creating .venv_sim
#
# It is a submodule, not a vendored copy: a URL and a pinned commit. Our own
# additions live in patches/GR00T-WholeBodyControl_sim_teleop.patch.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CHECK=0; VENV=1
for a in "$@"; do
  case "$a" in
    --check) CHECK=1 ;;
    --no-venv) VENV=0 ;;
    *) echo "unknown flag: $a" >&2; exit 2 ;;
  esac
done

D=external/GR00T-WholeBodyControl
PIN=b042411fae38ee4d1af9aac82a37a1f8d14d6dd0
URL=https://github.com/NVlabs/GR00T-WholeBodyControl.git
P="$ROOT/patches/GR00T-WholeBodyControl_sim_teleop.patch"
# The only large files the pipeline needs (Git LFS): the G1 model used by the
# controller and the sim, and the two whole-body-control policies.
LFS_INCLUDE="external_dependencies/unitree_sdk2_python/**,decoupled_wbc/control/robot_model/model_data/g1/meshes/*,decoupled_wbc/control/robot_model/model_data/g1/*.urdf,decoupled_wbc/sim2mujoco/resources/robots/g1/policy/*,gear_sonic/data/robot_model/model_data/g1/**"

have_checkout() { [ -f "$D/decoupled_wbc/control/main/teleop/run_g1_control_loop.py" ]; }
patched() { [ -f "$D/decoupled_wbc/control/sensor/vla_image_client.py" ]; }
lfs_ok() {   # a real ONNX file, not an LFS pointer
  local f="$D/decoupled_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-Balance.onnx"
  [ -f "$f" ] && ! head -c 40 "$f" | grep -q "git-lfs"
}

if [ "$CHECK" = 1 ]; then
  have_checkout && echo "  checkout   present ($(git -C "$D" rev-parse --short HEAD))" || echo "  checkout   MISSING"
  patched && echo "  patch      applied" || echo "  patch      not applied"
  lfs_ok && echo "  models     present" || echo "  models     MISSING (Git LFS)"
  [ -x "$D/.venv_sim/bin/python" ] && echo "  .venv_sim  present" || echo "  .venv_sim  MISSING"
  exit 0
fi

git lfs version >/dev/null 2>&1 \
  || { echo "git-lfs is required:  sudo apt install git-lfs && git lfs install" >&2; exit 1; }

# 1. the checkout, at the pinned commit, without downloading every LFS object
if ! have_checkout; then
  echo "[setup] fetching $URL @ ${PIN:0:10}"
  if git submodule status "$D" >/dev/null 2>&1; then
    GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --depth 1 "$D"
  else
    rm -rf "$D"
    GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --no-checkout "$URL" "$D"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$D" checkout --detach "$PIN"
  fi
fi
have_checkout || { echo "[setup] could not fetch $D" >&2; exit 1; }

# 2. only the model files we use (~55 MB instead of the full LFS set)
if ! lfs_ok; then
  echo "[setup] fetching the G1 model and WBC policies (Git LFS)"
  git -C "$D" lfs pull --include="$LFS_INCLUDE" || { echo "[setup] git lfs pull failed" >&2; exit 1; }
fi

# 3. our changes: the camera client the exporter records from, grasp smoothing,
#    the sim's DDS domain (99) and the 3-camera recording defaults
if patched; then
  echo "[setup] $(basename "$P") already applied"
elif git -C "$D" apply --check "$P" 2>/dev/null; then
  git -C "$D" apply "$P" && echo "[setup] applied $(basename "$P")"
else
  echo "[setup] cannot apply $(basename "$P") - is $D at ${PIN:0:10}?" >&2; exit 1
fi

# 4. the DDS environment the simulator runs in (unitree_sdk2py + gear_sonic),
#    built with NVIDIA's own installer
if [ "$VENV" = 1 ] && [ ! -x "$D/.venv_sim/bin/python" ]; then
  echo "[setup] building $D/.venv_sim (NVIDIA's install_scripts/install_mujoco_sim.sh)"
  (cd "$D" && bash install_scripts/install_mujoco_sim.sh) || { echo "[setup] .venv_sim failed" >&2; exit 1; }
fi

cat <<EOF

[setup] done.
  controller, teleop, exporter   $D/decoupled_wbc  (run in NVIDIA's container)
  DDS sim env                    $D/.venv_sim
  WBC policies                   $D/decoupled_wbc/sim2mujoco/resources/robots/g1/policy

Pull the container image once:
  cd $D && ./decoupled_wbc/docker/run_docker.sh --install --root
EOF
