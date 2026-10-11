#!/usr/bin/env bash
# Open the MuJoCo sim, the MuGS hybrid and the robot camera - one command.
#
#   scripts/realsim/start.sh              sim waits for YOUR whole-body controller
#   scripts/realsim/start.sh --hold       hold the initial pose instead of falling
#   scripts/realsim/start.sh --paused     start with physics stopped, to look
#                                         at a scene: a humanoid topples in a
#                                         second or two without a balancer, so
#                                         --hold alone is not enough to see it
#   scripts/realsim/start.sh --builtin    run the bundled controller in-process
#   scripts/realsim/start.sh --vla        also serve the VLA cameras on :5555
#   scripts/realsim/start.sh --fixed-layout   do NOT reshuffle the apple/basket
#   scripts/realsim/start.sh --layout-seed N  reproducible reshuffles
#   scripts/realsim/start.sh --relay      use wbc_state_relay instead of the
#                                         sim's own state publisher
#   scripts/realsim/start.sh --visual-dr [CONFIG]  new lighting / colours /
#                                         background look on every reset
#                                         (backspace); ranges in
#                                         configs/visual_randomization.yaml
#   scripts/realsim/start.sh --stop       close everything
#   scripts/realsim/start.sh --scene outputs/myroom --robot g1_cad
#
# --scene is required: it is the directory build_sim_scene.sh wrote.
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

# One place for machine-specific locations (configs/paths.env), loaded for us.
# shellcheck disable=SC1091
. "$ROOT/scripts/_bootstrap.sh" || exit 1
SIM="$RS_PY_SIM"
VENVSIM="${RS_PY_DDS:-}"
SCENE=""
ROBOT=g1_cad
CAMERA=cap
FOVY=""            # empty: let the viewer use the scene's own camera fovy
STATE_PORT=5601
CAM_STATE_PORT=5602
VLA_STATE_PORT=5603
CAM_PORT=5562
VLA_PORT=5555
# Wire format for the VLA cameras. The two consumers disagree:
#   multipart  VLAImageClient  - run_g1_data_exporter.py (teleop recording)
#   msgpack    SensorClient    - the VLA inference clients
# A SUB socket with a topic filter silently drops single-part msgpack,
# so the exporter reports "NO ZMQ frames received" with the wrong one.
VLA_PROTOCOL="${VLA_PROTOCOL:-multipart}"
VLA=0
RANDOMIZE=1
LAYOUT_SEED=
VLA_EGO=head_pinhole
DOMAIN=99
IFACE=lo
BUILTIN=0
HOLD=0
PAUSED=0
RELAY=0
VISUAL_DR=0
VISUAL_CFG=
LOGS=/tmp/realsim
CEIL="${RS_VRAM_CEILING_MIB:-31000}"

ours() {   # print the pids of everything this script starts, never this shell
  # Match the script paths, never a command line, so this cannot kill the shell
  # that invoked it.
  ps -eo pid,ppid,cmd > /tmp/realsim_stop.txt
  awk -v me="$$" -v pa="$PPID" '
        $1==me || $2==me || $1==pa { next }
        /run_mujoco_wbc_sim|wbc_state_relay|realsim\/run_wbc|realsim\/vla_camera_server|mugs_viewer|elp_camera_pub|elp_camera_sub|vla_camera_viewer/ && !/awk/ { print $1 }
      ' /tmp/realsim_stop.txt
}

stop_all() {   # print the pids it killed, non-zero if there was nothing
  local P
  P=$(ours)
  [ -n "$P" ] || return 1
  kill $P 2>/dev/null
  # WAIT for them, do not guess. MuJoCo takes a second or two to exit, and a
  # sim still dying holds DDS domain 99 and its UDP sockets: start on top of it
  # and you get two sims stepping, two state publishers, and tests that pass or
  # fail depending on timing. A fixed sleep was not enough - this is measured.
  local i
  for i in $(seq 60); do
    [ -z "$(ours)" ] && break
    sleep 0.25
  done
  [ -z "$(ours)" ] || echo "WARNING: still alive after 15s: $(ours)" >&2
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
    --vla)     VLA=1; shift ;;
    --fixed-layout) RANDOMIZE=0; shift ;;
    --layout-seed)  LAYOUT_SEED="$2"; shift 2 ;;
    --relay)   RELAY=1; shift ;;
    --vla-ego) VLA_EGO="$2"; shift 2 ;;
    --hold)    HOLD=1; shift ;;
    --paused)  PAUSED=1; shift ;;
    --visual-dr)
      VISUAL_DR=1; shift
      # optional config path right after the flag
      if [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; then VISUAL_CFG="$1"; shift; fi ;;
    --stop)
      if P=$(stop_all); then echo "stopped: $P"; else echo "nothing running"; fi
      exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$SCENE" ]; then
  {
    echo "need --scene <dir>: the directory scripts/build_sim_scene.sh wrote."
    if [ -n "$(ls -d outputs/*/mujoco 2>/dev/null)" ]; then
      echo; echo "Scenes you have:"
      for d in outputs/*/mujoco; do echo "    --scene $(dirname "$d")"; done
    else
      echo; echo "You have not built a scene yet - see README, section 3."
    fi
  } >&2
  exit 2
fi

# The hybrid window's framing is a property of the SCENE, not of this script.
# outputs/<scene>/realsim_view.json carries it when someone has tuned a view they
# like; without it the viewer auto-frames from the scene's bodies, which is what
# makes a freshly built scene work here with no edits.
VIEW_JSON="$SCENE/realsim_view.json"
VIEWARGS=(); FOVYARGS=()
if [ -f "$VIEW_JSON" ]; then
  while IFS= read -r line; do VIEWARGS+=("$line"); done < <(
    "$SIM" - "$VIEW_JSON" <<'PYEOF'
import json, sys
v = json.load(open(sys.argv[1]))
out = []
if "look_at" in v:   out += ["--look-at", *[f"{x}" for x in v["look_at"]]]
if "distance" in v:  out += ["--distance", f"{v['distance']}"]
if "azimuth" in v:   out += ["--azimuth", f"{v['azimuth']}"]
if "elevation" in v: out += ["--elevation", f"{v['elevation']}"]
if "pos" in v:       out += ["--pos", *[f"{x}" for x in v["pos"]]]
print("\n".join(out))
PYEOF
  )
  echo "[start] view from $VIEW_JSON"
else
  echo "[start] no realsim_view.json - the hybrid window will auto-frame"
fi
# --fovy on the command line wins; otherwise take the scene's if it set one.
if [ -z "$FOVY" ] && [ -f "$VIEW_JSON" ]; then
  FOVY=$("$SIM" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("fovy",""))' "$VIEW_JSON")
fi
# A static support surface (the table) does not need to be DRAWN - the splat
# already shows it, sharply. It only needs collision, so objects rest on it.
# Listing it here tells every renderer not to draw its mesh, which is what lets
# the photoreal table show through instead of a second, duller copy of it.
EXCLUDE=""
if [ -f "$VIEW_JSON" ]; then
  EXCLUDE=$("$SIM" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("exclude_geoms",""))' "$VIEW_JSON")
fi
EXARGS=(); [ -n "$EXCLUDE" ] && { EXARGS=(--exclude-geoms "$EXCLUDE"); echo "[start] not drawing: $EXCLUDE (the splat shows them)"; }

[ -n "$FOVY" ] && FOVYARGS=(--fovy "$FOVY")

RENDER_MJCF="$SCENE/mujoco/scene_g1.xml"
[ -f "$RENDER_MJCF" ] || RENDER_MJCF="$SCENE/mujoco/scene_robot.xml"
WBC_MJCF="$SCENE/mujoco/wbc/scene_wbc.xml"
PLY=$(ls -t "$SCENE"/gs_bg_notripod.ply "$SCENE"/gs_bg_clean.ply "$SCENE"/gs_bg.ply 2>/dev/null | head -1)
if [ ! -f "$RENDER_MJCF" ]; then
  {
    echo "No scene at $SCENE (expected $SCENE/mujoco/scene_g1.xml)."
    echo "Build it with scripts/build_sim_scene.sh - README, section 3."
    echo "Check the whole setup with:  scripts/doctor.sh"
  } >&2
  exit 1
fi
if [ -z "$PLY" ]; then
  {
    echo "No background splat under $SCENE (looked for gs_bg_notripod.ply,"
    echo "gs_bg_clean.ply, gs_bg.ply). The hybrid render has no room to"
    echo "composite against. Run the splat stages of build_sim_scene.sh:"
    echo "    scripts/build_sim_scene.sh --scene-dir $SCENE ... --from colmap"
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

# One UDP port per mirror consumer: two sockets cannot bind the same one and the
# second dies with "Address already in use". The VLA camera server is a third
# consumer, so it only gets a port when it is actually being started.
MIRROR_PORTS="$STATE_PORT,$CAM_STATE_PORT"
[ "$VLA" = 1 ] && MIRROR_PORTS="$MIRROR_PORTS,$VLA_STATE_PORT"
# An extra listener for the object stream, for anything that wants to know
# where the objects are without displacing one of the fixed consumers - each
# binds its own UDP port, so another process needs another port.
#   EXTRA_MIRROR_PORTS=5604 scripts/realsim/start.sh --vla --scene outputs/myroom
# Inert when unset.
MIRROR_PORTS="$MIRROR_PORTS${EXTRA_MIRROR_PORTS:+,$EXTRA_MIRROR_PORTS}"

# Visual domain randomization: the sim draws a seed per reset and broadcasts it
# on the object ports; each renderer below turns it into the same look.
VDR_SIM=(); VDR_REN=()
if [ "$VISUAL_DR" = 1 ]; then
  VDR_SIM=(--visual-dr)
  VDR_REN=(--visual-dr ${VISUAL_CFG:+"$VISUAL_CFG"})
  echo "[start] visual randomization on: ${VISUAL_CFG:-configs/visual_randomization.yaml}"
fi
VDR_VLA=("${VDR_REN[@]}")

# If a previous run died, its children outlive it: a second wbc_state_relay
# doubles the UDP state every window reads, and the camera publisher still owns
# CAM_PORT - that is the ZMQError "Address already in use (addr='tcp://*:5562')".
# Clear them before starting, rather than half-starting on top of them.
if LEFT=$(stop_all); then
  echo "[start] cleared leftovers from a previous run: $LEFT"
fi

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
NEED=6000; [ "$VLA" = 1 ] && NEED=9000   # the VLA server holds its own splat
echo "[start] VRAM ${USED} MiB used of a ${CEIL} MiB ceiling, this needs ~${NEED}"
if [ $((USED + NEED)) -gt "$CEIL" ]; then
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

LAYOUTARGS=()
[ "$RANDOMIZE" = 0 ] && LAYOUTARGS+=(--no-randomize)
[ -n "$LAYOUT_SEED" ] && LAYOUTARGS+=(--seed "$LAYOUT_SEED")

if [ "$BUILTIN" = 1 ]; then
  # Bundled controller, in-process: it owns the physics and broadcasts its own
  # pose, so no DDS and no relay are involved.
  echo "[start] scene $RENDER_MJCF  (bundled controller)"
  MUJOCO_GL=glfw setsid nohup "$SIM" scripts/realsim/run_wbc.py \
      --robot "$ROBOT" --mjcf "$RENDER_MJCF" --prefix g1_ --ply "$PLY" \
      --camera "$CAMERA" "${FOVYARGS[@]}" --width 540 --height 720 \
      --viewer --publish-state "$STATE_PORT" "${LAYOUTARGS[@]}" \
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
  [ "$PAUSED" = 1 ] && HOLDFLAG+=(--paused)
  echo "[start] sim scene    $WBC_MJCF"
  echo "[start] render scene $RENDER_MJCF"
  # What the sim streams to the photoreal windows, and why it is the sim doing it:
  #   --publish-objects  DDS carries the robot ONLY, so object poses (the apple,
  #                      the basket) exist in no topic a relay could observe.
  #                      Without this every window shows them frozen at their
  #                      spawn pose while the sim's have moved.
  #   --publish-state    the robot, in the same 51-float format the relay uses.
  #                      This process IS the DDS bridge, so mj_data is already
  #                      here and a separate relay is pure indirection - plus a
  #                      second process to leak, which is how two relays once
  #                      ended up doubling every window's state. --relay puts
  #                      the observer back for a sim we do not own.
  # NOTE: keep these comments ABOVE the env-var line. A comment after a trailing
  # backslash swallows the continuation and the command silently loses its env.
  SIMPUB=(--publish-objects "$MIRROR_PORTS")
  # Objects get a fresh, reachable spot on start and on every viewer reset
  # (backspace) so collected episodes are not all the same layout.
  [ "$RANDOMIZE" = 0 ] && SIMPUB+=(--no-randomize)
  [ -n "$LAYOUT_SEED" ] && SIMPUB+=(--seed "$LAYOUT_SEED")
  [ "$RELAY" = 0 ] && SIMPUB+=(--publish-state "$MIRROR_PORTS")
  PYTHONPATH="$RS_DEPLOY_ROOT" MUJOCO_GL=glfw \
  setsid nohup "$VENVSIM" scripts/run_mujoco_wbc_sim.py \
      --scene-dir "$SCENE" --mjcf "$WBC_MJCF" "${HOLDFLAG[@]}" \
      "${SIMPUB[@]}" "${VDR_SIM[@]}" \
      > "$LOGS/sim.log" 2>&1 < /dev/null & disown
  wait_for "$LOGS/sim.log" "WAITING FOR WBC|rt/lowcmd" "mujoco sim" || exit 1

  if [ "$RELAY" = 1 ]; then
    setsid nohup "$VENVSIM" scripts/wbc_state_relay.py \
        --domain "$DOMAIN" --interface "$IFACE" \
        --port "$MIRROR_PORTS" --rate 50 \
        > "$LOGS/relay.log" 2>&1 < /dev/null & disown
    wait_for "$LOGS/relay.log" "." "state relay" || exit 1
  else
    wait_for "$LOGS/sim.log" "\\[state\\] publishing" "sim state publisher" || exit 1
  fi
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
      --camera scene_view --capture-camera "$CAMERA" "${FOVYARGS[@]}" \
      --width 700 --height 933 "${VIEWARGS[@]}" "${EXARGS[@]}" \
      --mirror "$STATE_PORT" --mirror-frame "$MIRROR_FRAME" "${VDR_REN[@]}" \
      > "$LOGS/hybrid.log" 2>&1 < /dev/null & disown
  wait_for "$LOGS/hybrid.log" "mirror|Gaussian" "mugs hybrid" || exit 1
fi

# The robot's camera, following the same state as everything else.
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl setsid nohup "$SIM" scripts/elp_camera_publisher.py \
    --scene-dir "$SCENE" --mjcf "$RENDER_MJCF" --ply "$PLY" \
    --port "$CAM_PORT" --width 768 --height 432 --projection rectilinear --fps 12 \
    "${EXARGS[@]}" \
    --mirror "$CAM_STATE" --mirror-frame "$MIRROR_FRAME" "${VDR_REN[@]}" \
    > "$LOGS/camera_pub.log" 2>&1 < /dev/null & disown
wait_for "$LOGS/camera_pub.log" "PUB tcp" "robot camera" || exit 1

# The VLA camera server: the three cameras gr00t_wbc's data exporter and its
# inference client read, rendered photoreal from THIS scene. Without it the
# exporter sits on "image False" forever, because run_mujoco_wbc_sim.py
# publishes no images at all - the ELP pair is a different protocol and a
# different pair of topics.
if [ "$VLA" = 1 ]; then
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl setsid nohup "$SIM" scripts/realsim/vla_camera_server.py \
      --scene-dir "$SCENE" --mjcf "$RENDER_MJCF" --ply "$PLY" \
      --port "$VLA_PORT" --ego-camera "$VLA_EGO" --fps 20 "${EXARGS[@]}" \
      --protocol "$VLA_PROTOCOL" \
      --mirror "$VLA_STATE_PORT" --mirror-frame "$MIRROR_FRAME" "${VDR_VLA[@]}" \
      > "$LOGS/vla_cam.log" 2>&1 < /dev/null & disown
  wait_for "$LOGS/vla_cam.log" "PUB tcp" "vla cameras" || exit 1
fi

setsid nohup "$SIM" scripts/elp_camera_subscriber.py --port "$CAM_PORT" \
    --side both --topmost --pos-left 20 60 --pos-right 700 60 \
    > "$LOGS/camera_sub.log" 2>&1 < /dev/null & disown

# The wrist cameras, from the VLA server - the same frames the policy receives.
if [ "$VLA" = 1 ]; then
  setsid nohup "$SIM" scripts/realsim/vla_camera_viewer.py --port "$VLA_PORT" \
      --topics hand_left,hand_right --topmost --pos "20,560;700,560" \
      > "$LOGS/wrist_view.log" 2>&1 < /dev/null & disown
fi

echo
echo "[start] windows are up on $DISPLAY. Logs under $LOGS/"
if [ "$BUILTIN" = 0 ]; then
cat <<EOF

The sim is WAITING FOR WBC and the robot is falling. Start your controller:

  docker exec -it -w /root/Projects/GR00T-WholeBodyControl \\
      -e PYTHONPATH=/root/Projects/GR00T-WholeBodyControl decoupled_wbc-bash-root bash -c '
    source /root/venv/bin/activate; source /opt/ros/humble/setup.bash
    python decoupled_wbc/control/main/teleop/run_g1_control_loop.py \\
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
  PYTHONPATH=...   makes the MOUNTED checkout (with our patch: DOMAIN_ID 99,
                   the VLA camera client) win over the copy baked into the
                   image, whose config says DOMAIN_ID: 0 - the real robot. Our
                   sim is on 99, so it would connect to nothing, silently.

(container not up:  cd $RS_DEPLOY_ROOT && ./decoupled_wbc/docker/run_docker.sh --root
 - it opens a shell in the container; run the command above from another terminal)
EOF
fi
echo
if [ "$VLA" = 1 ]; then
  echo "VLA cameras are on tcp://*:$VLA_PORT (topics stereo/right, hand_left,"
  echo "hand_right, plus gear_sonic's msgpack keys). Record with:"
  echo "  run_g1_data_exporter.py --camera_host localhost --camera_port $VLA_PORT"
  echo "See docs/teleop_dataset.md - recording is keys c / x on the ROS"
  echo "topic /Gr00tKeyboardListener."
else
  echo "For data collection add --vla: it serves the three cameras the gr00t_wbc"
  echo "exporter reads, rendered photoreal from this scene. Nothing else"
  echo "publishes them, so the exporter otherwise records 'image False' forever."
fi
echo
echo "  tail -f $LOGS/sim.log"
echo "  scripts/realsim/start.sh --stop"
