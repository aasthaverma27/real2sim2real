#!/usr/bin/env bash
# Open the MuJoCo sim, the MuGS hybrid and the robot camera - one command.
#
#   scripts/realsim/start.sh              sim waits for YOUR whole-body controller
#   scripts/realsim/start.sh --hold       hold the initial pose instead of falling
#   scripts/realsim/start.sh --builtin    run the bundled controller in-process
#   scripts/realsim/start.sh --stop       close everything
#   scripts/realsim/start.sh --scene outputs/other --robot myrobot
#
# DEFAULT (no --builtin): the sim is the DDS one. It prints WAITING FOR WBC and
# the robot falls under gravity until a controller starts publishing rt/lowcmd -
# the same behaviour as the reference stack. Start your controller whenever you
# like; the sim picks it up mid-run.
#
# Why a relay is in the middle: no environment here can do both jobs.
#   .venv_sim      unitree_sdk2py yes, gsplat no   - talks DDS, cannot draw splats
#   robosnap-sim   unitree_sdk2py no,  gsplat yes  - draws splats, cannot talk DDS
# So wbc_state_relay.py watches the DDS traffic from .venv_sim and forwards the
# robot's pose as plain UDP, which the rendering environment can read.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# One place for machine-specific locations, loaded for us.
. "$ROOT/scripts/_bootstrap.sh"
SIM="$RS_PY_SIM"
VENVSIM="${RS_PY_DDS:-}"
SCENE=outputs/g1_ok
ROBOT=g1_29dof_with_hand
CAMERA=cap
FOVY=55.21
STATE_PORT=5601
CAM_STATE_PORT=5602
CAM_PORT=5562
DOMAIN=99
IFACE=lo
BUILTIN=0
HOLD=0
LOGS=/tmp/realsim
export DISPLAY="${DISPLAY:-:1}"
export PATH="$(dirname "$SIM"):$PATH"

stop_all() {   # print the pids it killed, non-zero if there was nothing
  # Match the script paths, never a command line, so this cannot kill the shell
  # that invoked it.
  ps -eo pid,ppid,cmd > /tmp/realsim_stop.txt
  local P
  P=$(awk -v me="$$" -v pa="$PPID" '
        $1==me || $2==me || $1==pa { next }
        /run_mujoco_wbc_sim|wbc_state_relay|realsim\/run_wbc|mugs_viewer|elp_camera_pub|elp_camera_sub/ && !/awk/ { print $1 }
      ' /tmp/realsim_stop.txt)
  [ -n "$P" ] || return 1
  kill $P 2>/dev/null
  echo $P
  return 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --scene)   SCENE="$2"; shift 2 ;;
    --robot)   ROBOT="$2"; shift 2 ;;
    --camera)  CAMERA="$2"; shift 2 ;;
    --fovy)    FOVY="$2"; shift 2 ;;
    --builtin) BUILTIN=1; shift ;;
    --hold)    HOLD=1; shift ;;
    --stop)
      if P=$(stop_all); then echo "stopped: $P"; else echo "nothing running"; fi
      exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

RENDER_MJCF="$SCENE/mujoco/scene_g1.xml"
[ -f "$RENDER_MJCF" ] || RENDER_MJCF="$SCENE/mujoco/scene_robot.xml"
WBC_MJCF="$SCENE/mujoco/wbc/scene_wbc.xml"
PLY=$(ls -t "$SCENE"/gs_bg_notripod.ply "$SCENE"/gs_bg_clean.ply "$SCENE"/gs_bg.ply 2>/dev/null | head -1)
if [ ! -f "$RENDER_MJCF" ]; then
  {
    echo "No scene at $SCENE."
    echo
    echo "Expected one of:"
    echo "    $SCENE/mujoco/scene_robot.xml"
    echo "    $SCENE/mujoco/scene_g1.xml"
    if [ -d outputs ] && [ -n "$(ls -d outputs/*/mujoco 2>/dev/null)" ]; then
      echo
      echo "Scenes you do have:"
      for d in outputs/*/mujoco; do echo "    --scene $(dirname "$d")"; done
    else
      echo
      echo "You have not built a scene yet. See README section 2,"
      echo "'Build your first scene' - roughly 20 minutes from an image."
    fi
    echo
    echo "Check the whole setup with:  scripts/doctor.sh"
  } >&2
  exit 1
fi
if [ -z "$PLY" ]; then
  {
    echo "No background splat under $SCENE (looked for gs_bg_notripod.ply,"
    echo "gs_bg_clean.ply, gs_bg.ply)."
    echo
    echo "The sim and the robot cameras work without one, but the hybrid render"
    echo "has no room to composite against. Build it: README section 4.4."
  } >&2
  exit 1
fi
if [ "$BUILTIN" = 0 ] && [ ! -f "$WBC_MJCF" ]; then
  echo "missing $WBC_MJCF - build it first:" >&2
  echo "  $SIM scripts/realsim/build_wbc_scene.py --scene-dir $SCENE \\" >&2
  echo "      --robot $ROBOT --scene-xml $RENDER_MJCF --strip-prefix g1_ \\" >&2
  echo "      --at <X> <Y> --yaw <RAD>" >&2
  exit 1
fi
mkdir -p "$LOGS"

# If a previous run died, its children outlive it: a second wbc_state_relay
# doubles the UDP state every window reads, and the camera publisher still owns
# CAM_PORT - that is the ZMQError "Address already in use (addr='tcp://*:5562')".
# Clear them before starting, rather than half-starting on top of them.
if LEFT=$(stop_all); then
  echo "[start] cleared leftovers from a previous run: $LEFT"
  sleep 2
fi

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
CEIL="${RS_VRAM_CEILING_MIB:-31000}"
echo "[start] VRAM ${USED} MiB used of a ${CEIL} MiB ceiling, this needs ~6000"
if [ $((USED + 6000)) -gt "$CEIL" ]; then
  echo "[start] ABORT: that would cross the ceiling. Holding the card:" >&2
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader >&2
  exit 1
fi

wait_for() {   # wait_for <log> <pattern> <label>
  echo -n "[start] $3 "
  for _ in $(seq 100); do
    grep -qE "$2" "$1" 2>/dev/null && { echo " up"; return 0; }
    grep -qiE "Traceback|ValueError|ImportError" "$1" 2>/dev/null && {
        echo " FAILED"; tail -20 "$1" >&2; return 1; }
    echo -n .; sleep 2
  done
  echo " TIMEOUT"; tail -20 "$1" >&2; return 1
}

if [ "$BUILTIN" = 1 ]; then
  # Bundled controller, in-process: it owns the physics and broadcasts its own
  # pose, so no DDS and no relay are involved.
  echo "[start] scene $RENDER_MJCF  (bundled controller)"
  MUJOCO_GL=glfw setsid nohup "$SIM" scripts/realsim/run_wbc.py \
      --robot "$ROBOT" --mjcf "$RENDER_MJCF" --prefix g1_ --ply "$PLY" \
      --camera "$CAMERA" --fovy "$FOVY" --width 540 --height 720 \
      --viewer --publish-state "$STATE_PORT" \
      > "$LOGS/wbc.log" 2>&1 < /dev/null & disown
  wait_for "$LOGS/wbc.log" "publishing state" "sim + hybrid" || exit 1
  MIRROR_FRAME=identity
else
  # The DDS sim: robot falls until YOUR controller publishes rt/lowcmd.
  if [ -z "$VENVSIM" ] || [ ! -x "$VENVSIM" ]; then
    echo "[start] no DDS interpreter (RS_PY_DDS=${VENVSIM:-unset})." >&2
    echo "[start] That environment provides gear_sonic + unitree_sdk2py and is" >&2
    echo "[start] only needed to drive a DEPLOYED controller. For the bundled" >&2
    echo "[start] controller instead:  scripts/realsim/start.sh --builtin" >&2
    exit 1
  fi
  HOLDFLAG=(--no-hold); [ "$HOLD" = 1 ] && HOLDFLAG=()
  echo "[start] sim scene    $WBC_MJCF"
  echo "[start] render scene $RENDER_MJCF"
  PYTHONPATH="$RS_DEPLOY_ROOT" MUJOCO_GL=glfw \
  # --publish-objects: DDS carries the robot only, so the sim streams the pose of
  # every free joint in the scene - the apple, the basket - to the same ports the
  # relay uses. Without it every photoreal window shows those objects frozen at
  # their spawn pose while the sim's have moved, which is exactly the visual gap
  # a recorded dataset must not have.
  setsid nohup "$VENVSIM" scripts/run_mujoco_wbc_sim.py \
      --scene-dir "$SCENE" --mjcf "$WBC_MJCF" "${HOLDFLAG[@]}" \
      --publish-objects "$STATE_PORT,$CAM_STATE_PORT" \
      > "$LOGS/sim.log" 2>&1 < /dev/null & disown
  wait_for "$LOGS/sim.log" "WAITING FOR WBC|rt/lowcmd" "mujoco sim" || exit 1

  setsid nohup "$VENVSIM" scripts/wbc_state_relay.py \
      --domain "$DOMAIN" --interface "$IFACE" \
      --port "$STATE_PORT,$CAM_STATE_PORT" --rate 50 \
      > "$LOGS/relay.log" 2>&1 < /dev/null & disown
  wait_for "$LOGS/relay.log" "." "state relay" || exit 1
  MIRROR_FRAME=auto
fi

# With --builtin there is no relay and the in-process publisher sends to one
# port only, so both consumers share it; with the relay each gets its own.
if [ "$BUILTIN" = 1 ]; then CAM_STATE="$STATE_PORT"; else CAM_STATE="$CAM_STATE_PORT"; fi

echo "[start] splat $PLY"

# The MuGS hybrid. With --builtin it follows the in-process controller; otherwise
# it follows the DDS sim through the relay, undoing build_wbc_scene's re-origin.
if [ "$BUILTIN" = 0 ]; then
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl setsid nohup "$SIM" scripts/mugs_viewer.py \
      --scene-dir "$SCENE" --mjcf "$RENDER_MJCF" --ply "$PLY" \
      --camera scene_view --capture-camera "$CAMERA" --fovy "$FOVY" \
      --width 700 --height 933 \
      --look-at 0.0216 0.8297 0.6614 --distance 1.97 \
      --azimuth -93.6502 --elevation 25.1445 \
      --pos 0 30 \
      --mirror "$STATE_PORT" --mirror-frame "$MIRROR_FRAME" \
      > "$LOGS/hybrid.log" 2>&1 < /dev/null & disown
  wait_for "$LOGS/hybrid.log" "mirror|Gaussian" "mugs hybrid" || exit 1
fi

# The robot's camera, following the same state as everything else.
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl setsid nohup "$SIM" scripts/elp_camera_publisher.py \
    --scene-dir "$SCENE" --mjcf "$RENDER_MJCF" --ply "$PLY" \
    --port "$CAM_PORT" --width 640 --height 480 --fov-h 170 --face-size 640 --fps 6 \
    --mirror "$CAM_STATE" --mirror-frame "$MIRROR_FRAME" \
    > "$LOGS/camera_pub.log" 2>&1 < /dev/null & disown
wait_for "$LOGS/camera_pub.log" "PUB tcp" "robot camera" || exit 1

setsid nohup "$SIM" scripts/elp_camera_subscriber.py --port "$CAM_PORT" \
    --side both --topmost --pos-left 20 60 --pos-right 700 60 \
    > "$LOGS/camera_sub.log" 2>&1 < /dev/null & disown

echo
echo "[start] windows are up on $DISPLAY. Logs under $LOGS/"
if [ "$BUILTIN" = 0 ]; then
cat <<EOF

The sim is WAITING FOR WBC and the robot is falling. Start your controller:

  docker exec -it -w /root/Projects/deploy-wbc-on-robot \\
      -e PYTHONPATH=/root/Projects/deploy-wbc-on-robot gr00t_wbc-deploy-root bash -c '
    source /root/venv/bin/activate; source /opt/ros/humble/setup.bash
    python gr00t_wbc/control/main/teleop/run_g1_control_loop.py \\
      --wbc_version gear_wbc \\
      --wbc_model_path policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx \\
      --wbc_policy_class G1DecoupledWholeBodyPolicy \\
      --interface lo --simulator None --control_frequency 50 \\
      --no-enable_waist --with_hands --no-high_elbow_pose --no-enable_gravity_compensation'

then press ] to activate it. Confirm 'Use policy action: True'.

Three things in that command are load-bearing:
  -it              the default keyboard dispatcher reads stdin; with no TTY the
                   ] keypress never lands and use_policy_action stays False
  source ...       docker exec skips /root/.bashrc, which is what activates the
                   venv and ROS - without it: ModuleNotFoundError: rclpy
  PYTHONPATH=...   the editable install points gr00t_wbc at the IMAGE's
                   /root/Projects/GR00T-WholeBodyControl tree, whose
                   g1_29dof_gear_wbc.yaml says DOMAIN_ID: 0. Our sim is on 99,
                   so it would connect to nothing, silently. The mounted deploy
                   tree says 99; this makes it win.

(container not Up:  cd $RS_DEPLOY_ROOT/deploy-wbc-on-robot
 && ./docker/run_docker.sh --deploy --root)
EOF
fi
echo
echo "  tail -f $LOGS/sim.log"
echo "  scripts/realsim/start.sh --stop"
