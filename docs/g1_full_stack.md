# The four-process stack, with the photoreal windows attached

Two different things get called "running the WBC" and they are not the same:

| | what it is | photoreal? | real WBC? |
|---|---|---|---|
| **A. In-process harness** | `play_groot_wbc.py` — sim + WBC ONNX + MuGS in ONE process | yes | the shipped ONNX nets, driven from MuJoCo state |
| **B. Four-process stack** | policy server -> sim -> WBC container -> client, over DDS | yes, via the UDP mirror | the real deployed controller, over DDS |

**A** is the one command in `HANDOVER_g1_apple.md` §4.2 and it is what you should
use to look at the scene. **B** is the stack you remembered, and it is below.

---

## Why B needs a mirror at all

No single environment here can both talk DDS and rasterise gaussians:

```
/mnt/ssd2/vla_for_simulation_testing/.venv_sim   unitree_sdk2py OK  gear_sonic OK  gsplat --
/mnt/ssd2/conda/envs/robosnap-sim                unitree_sdk2py --  gear_sonic --  gsplat OK
```

So `scripts/wbc_state_relay.py` runs in `.venv_sim`, observes the same DDS
topics the WBC observes, and forwards the robot's pose as fixed-size UDP
datagrams that anything can read with the standard library:

```
one datagram = 51 float32 = [tick, base_pos(3), base_quat_wxyz(4),
                             body_q(29), hand_q(14)]      joints in model order
```

It publishes nothing to DDS — a pure observer, invisible to the sim and the WBC.
`mugs_viewer.py --mirror`, `elp_camera_publisher.py --mirror` and
`hybrid_viewer.py --mirror` all consume it, so every photoreal window shows the
same robot the WBC is actually driving instead of stepping its own physics.

### The objects are a second stream

The robot is not the whole scene, and DDS carries only the robot: `rt/lowstate`
is motor state, `rt/odostate` is the base. Anything with a free joint — the
apple, the basket — is moved by the sim's physics and appears in no topic. A
mirroring renderer does not `mj_step`, so its own copy of those objects stays at
the spawn pose: push the apple in the sim viewer and the hybrid, the robot's
camera and every recorded frame keep showing it where it started. That is a
visual gap baked straight into a dataset.

So `run_mujoco_wbc_sim.py --publish-objects 5601,5602` streams them itself, to
the same ports, as a second kind of datagram:

```
object datagram = b"RSOB" + uint32 n + n * (char[40] name + 7 float32 [pos, quat_wxyz])
```

`MirrorReceiver.apply()` drains the socket keeping the last packet **of each
kind**, so the two senders coexist on one port. Objects are matched by name
after dropping `build_wbc_scene`'s `rs_` prefix — the sim's
`rs_obj1_red_apple_free` writes the renderer's `obj1_red_apple_free` — so joint
order does not matter and an object either side lacks is skipped with one
warning. Object poses get the same re-origin transform as the base, which is
what keeps them registered to the splat.

The robot's own base is excluded on purpose: it already arrives over DDS, and
publishing it twice would leave two writers fighting over one `qpos` slot.
`start.sh` passes the flag for you. `mugs_viewer.py --no-mirror-objects` opts
out, for placement work where the sim overwriting a dragged object is a
nuisance.

Verified by dropping the apple 30 cm in the sim and watching the renderer's copy
follow it down — 1.079 → 0.887 over one second, settling at 0.785.

---

## The processes, in start order

Terminal-by-terminal. Check the card first — the ceiling is 31 GB and it is
shared:

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader
```

### 1. GR00T policy server — host, ~7.8 GB, port 5560

```bash
cd /mnt/ssd2/Isaac-GR00T
export NO_ALBUMENTATIONS_UPDATE=1 HF_HOME=/mnt/ssd2/huggingface_cache \
       PYTHONPATH=/mnt/ssd2/Isaac-GR00T
setsid nohup /mnt/ssd2/dsrl/.venv/bin/python -u -m gr00t.eval.run_gr00t_server \
  --model_path /mnt/ssd2/Isaac-GR00T/checkpoints/red_green_blue_cube_sorting_in_sim/checkpoint-22000 \
  --embodiment_tag unitree_g1_full_body_with_waist_height_nav_cmd \
  --device cuda --host 0.0.0.0 --port 5560 \
  > /tmp/gr00t_server.log 2>&1 < /dev/null & disown

grep -m1 "Server ready" <(tail -f /tmp/gr00t_server.log)      # ~30 s
```

### 2. The sim — host, `.venv_sim`, DDS domain 99 on `lo`

This is the one that carries **our photoreal scene** and opens the MuJoCo
viewer. It reuses gear_sonic's `UnitreeSdk2Bridge` rather than reimplementing
the protocol:

```bash
cd /mnt/ssd2/robosnap
PYTHONPATH=/mnt/ssd2/vla_for_simulation_testing:$PYTHONPATH \
MUJOCO_GL=glfw DISPLAY=:1 \
/mnt/ssd2/vla_for_simulation_testing/.venv_sim/bin/python \
    scripts/run_mujoco_wbc_sim.py \
    --scene-dir outputs/g1_ok \
    --mjcf outputs/g1_ok/mujoco/scene_g1.xml
```

`--no-hold` stops it holding the initial pose while it waits for the WBC;
`--paused` starts paused. The flow it implements:

```
WBC --rt/lowcmd--> UnitreeSdk2Bridge --> PD --> torque --> MuJoCo
                                                             |
MuJoCo --rt/lowstate, rt/odostate, IMU--------------------> WBC
```

gear_sonic's own sim (`gear_sonic/scripts/run_sim_loop.py`, same venv) is the
alternative, but it brings its own scene — not the apple room.

### 3. The WBC — inside the Docker container

```bash
cd /mnt/ssd2/vla_for_simulation_testing/deploy-wbc-on-robot
./docker/run_docker.sh --deploy --root
```

Then **inside the container**:

```bash
export GR00T_WBC_TMUX_SESSION=g1_deployment && \
PYTHONPATH=/root/Projects/deploy-wbc-on-robot \
/root/venv/bin/python \
  /root/Projects/deploy-wbc-on-robot/gr00t_wbc/control/main/teleop/run_g1_control_loop.py \
  --wbc_version gear_wbc \
  --wbc_model_path policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx \
  --wbc_policy_class G1DecoupledWholeBodyPolicy \
  --interface lo \
  --simulator None \
  --control_frequency 50 \
  --no-enable_waist \
  --with_hands \
  --no-high_elbow_pose \
  --no-enable_gravity_compensation
```

Or in one shot from the host, no tmux and no interactive shell needed first —
this is the form that is known to work:

```bash
docker exec -it -w /root/Projects/deploy-wbc-on-robot \
    -e PYTHONPATH=/root/Projects/deploy-wbc-on-robot gr00t_wbc-deploy-root bash -c '
  source /root/venv/bin/activate; source /opt/ros/humble/setup.bash
  python gr00t_wbc/control/main/teleop/run_g1_control_loop.py \
    --wbc_version gear_wbc \
    --wbc_model_path policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx \
    --wbc_policy_class G1DecoupledWholeBodyPolicy \
    --interface lo --simulator None --control_frequency 50 \
    --no-enable_waist --with_hands --no-high_elbow_pose --no-enable_gravity_compensation'
```

Three load-bearing details, each of which fails in its own quiet way:

- **`-it`.** `keyboard_dispatcher_type` defaults to `raw`, i.e. stdin. With no
  TTY the listener dies with `Inappropriate ioctl for device`, `]` never lands,
  and `use_policy_action` stays `False`. Headless alternative:
  `--keyboard_dispatcher_type ros`, then `ros2 topic pub --once /keyboard_input
  std_msgs/msg/String "{data: ']'}"`.
- **the two `source` lines.** `docker exec` does not read `/root/.bashrc`, and
  that file is what activates `/root/venv` and `/opt/ros/humble`. Skip them and
  the import of `gr00t_wbc.control.envs.g1.g1_env` dies on
  `ModuleNotFoundError: No module named 'rclpy'`.
- **`PYTHONPATH`.** `__editable__.gr00t_wbc-0.1.0.pth` resolves `gr00t_wbc` to
  the **image's** `/root/Projects/GR00T-WholeBodyControl` tree, not the mounted
  `deploy-wbc-on-robot` one, and the two differ where it matters:
  that tree's `g1_29dof_gear_wbc.yaml` carries `DOMAIN_ID: 0` while our sim is
  on **99**, and it has no `control/sensor/vla_image_client.py`. A sys.path
  entry beats the editable finder, so this pins the right tree. Symptom without
  it: the WBC starts clean and the sim prints `WAITING FOR WBC` forever.

`--interface lo` is what makes it talk to the sim on loopback. The DDS domain
is 99 on both sides: `run_mujoco_wbc_sim.py` hardcodes it, `wbc_state_relay.py`
defaults to it, and the mounted tree's yaml says it. `selected interface "lo" is
not multicast-capable: disabling multicast` on startup is normal for loopback.

`--interface lo` is what makes it talk to the sim on loopback instead of a real
robot. Image `gr00t_wbc-deploy-root:latest` exists locally; the container
`gr00t_wbc-bash-root` is currently `Exited (137)`, so it needs restarting.
`docker/kill_all_containers.sh` and `docker/kill_gr00t_wbc_processors.sh` clean
up.

The lighter alternative, no Docker, is `simple_wbc_code` with keyboard teleop —
same two ONNX nets:

```bash
conda activate unitree_wbc_v2
cd /mnt/ssd2/simple_wbc_code
pip install -e external_dependencies/unitree_sdk2_python      # ONE TIME - see below
python ./wbc_policy_teleop.py
```

Controls: `]` activate walking, `O` deactivate, `WASD` translate, `Q`/`E` yaw,
`Z`/`SPACE` stop, `1`/`2` base height, `N`/`M` gait frequency.

### 4. The inference client — inside the container

```bash
/root/venv/bin/python \
  /root/Projects/deploy-wbc-on-robot/run_inference_smooth.py
```

It pulls actions from the policy server and publishes them on
`CONTROL_GOAL_TOPIC` over ROS2, and reads images through
`sdk.create_observation_reader`. Smoothing knobs:
`--smooth.ema-alpha 0.30`, `--smooth.max-delta-rad 0.06`,
`--smooth.chunk-blend-steps 8`.

### 5. The state relay — host, `.venv_sim`, UDP 5601

```bash
cd /mnt/ssd2/robosnap
setsid nohup /mnt/ssd2/vla_for_simulation_testing/.venv_sim/bin/python \
    scripts/wbc_state_relay.py --domain 99 --interface lo --port 5601 --rate 50 \
    > /tmp/relay.log 2>&1 < /dev/null & disown
tail -3 /tmp/relay.log
```

### 6. The MuGS hybrid window — host, `robosnap-sim`

```bash
cd /mnt/ssd2/robosnap
export PATH=/mnt/ssd2/conda/envs/robosnap-sim/bin:$PATH
SIM=/mnt/ssd2/conda/envs/robosnap-sim/bin/python

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl DISPLAY=:1 setsid nohup $SIM scripts/mugs_viewer.py \
    --scene-dir outputs/g1_ok \
    --mjcf outputs/g1_ok/mujoco/scene_g1.xml \
    --ply outputs/g1_ok/gs_bg_notripod.ply \
    --camera scene_view --capture-camera cap --fovy 55.21 \
    --width 700 --height 933 \
    --look-at 0.0216 0.8297 0.6614 --distance 1.97 \
    --azimuth -93.6502 --elevation 25.1445 \
    --splat-geoms round_wooden_table --pos 0 30 \
    --mirror 5601 \
    > /tmp/hybrid.log 2>&1 < /dev/null & disown
```

### 7. The robot's fisheye pair — host, `robosnap-sim`

Note `--port 5562`: 5560 is the policy server and 5555 is the camera server.

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl setsid nohup $SIM scripts/elp_camera_publisher.py \
    --scene-dir outputs/g1_ok \
    --mjcf outputs/g1_ok/mujoco/scene_g1.xml \
    --ply outputs/g1_ok/gs_bg_notripod.ply \
    --port 5562 --width 640 --height 480 --fov-h 170 --face-size 640 --fps 6 \
    --mirror 5601 \
    > /tmp/elp_pub.log 2>&1 < /dev/null & disown

# wait for "PUB tcp" in /tmp/elp_pub.log, then
DISPLAY=:1 setsid nohup $SIM scripts/elp_camera_subscriber.py --port 5562 \
    --side both --topmost --pos-left 20 60 --pos-right 700 60 \
    > /tmp/elp_sub.log 2>&1 < /dev/null & disown

tail -2 /tmp/elp_sub.log        # "pair synchronous, stamp delta 0.000 ms"
```

---

## Two gaps you will hit, stated up front

**1. `unitree_wbc_v2` is missing the SDK.** The conda env exists with mujoco,
onnxruntime, torch and pinocchio, but `unitree_sdk2py` is not installed in it.
The one-time `pip install -e external_dependencies/unitree_sdk2_python` in step
3 is required before `wbc_policy_teleop.py` will run. `.venv_sim` already has it.

**2. The client's camera topics do not match ours.** `run_inference_smooth.py`
reads four VLA cameras — `cam_left_high`, `cam_right_high`, `cam_left_wrist`,
`cam_right_wrist` — from a camera ZMQ server on port **5555**, mapped from
topics `stereo/left`, `stereo/right`, `hand_left`, `hand_right`.
`scripts/run_mujoco_wbc_sim.py` publishes **no** camera stream, and our ELP
publisher emits `elp/left` / `elp/right`. So step 4 has nothing to read from
unless you either run gear_sonic's camera server or bridge the topic names.
Steps 1-3 and 5-7 work without it: the WBC balances and every photoreal window
follows the live sim; only the policy's arm motion is missing.

Until that bridge exists, route **A** (`HANDOVER_g1_apple.md` §4.3) is the way
to get the checkpoint onto the arms in the photoreal scene, because it renders
the ego view in-process and hands it straight to the policy server.

---

## Stopping it all

```bash
pkill -f wbc_state_relay
pkill -f elp_camera
pkill -f mugs_viewer
pkill -f run_mujoco_wbc_sim
pkill -f run_gr00t_server
/mnt/ssd2/vla_for_simulation_testing/deploy-wbc-on-robot/docker/kill_all_containers.sh
```

Do not `pkill -f <pattern>` from a shell whose own command line contains that
pattern — it matches itself and kills your shell. Quote it as `'elp_camera_pub[l]isher'`.
