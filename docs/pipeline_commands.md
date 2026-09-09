# The pipeline, in start order

The dsrl reference stack, pointed at the photoreal apple scene. Four steps, in
this order. Each block is one terminal.

```bash
cd /mnt/ssd2/robosnap
SIM=/mnt/ssd2/conda/envs/robosnap-sim/bin/python
VENVSIM=/mnt/ssd2/vla_for_simulation_testing/.venv_sim/bin/python
export DISPLAY=:1
export PATH=/mnt/ssd2/conda/envs/robosnap-sim/bin:$PATH
```

Check the card first — 31 GB ceiling, and it is shared:

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader
```

Budget: policy server 7.8 GB + hybrid ~1.5 GB + camera ~1.5 GB + sim ~0.5 GB ≈ **12 GB**.

---

## 1. GR00T policy server

Same as dsrl. Host, `dsrl/.venv`, port 5560, ~7.8 GB, ~30 s to load.

```bash
cd /mnt/ssd2/Isaac-GR00T
export NO_ALBUMENTATIONS_UPDATE=1 HF_HOME=/mnt/ssd2/huggingface_cache \
       PYTHONPATH=/mnt/ssd2/Isaac-GR00T
setsid nohup /mnt/ssd2/dsrl/.venv/bin/python -u -m gr00t.eval.run_gr00t_server \
  --model_path /mnt/ssd2/Isaac-GR00T/checkpoints/red_green_blue_cube_sorting_in_sim/checkpoint-22000 \
  --embodiment_tag unitree_g1_full_body_with_waist_height_nav_cmd \
  --device cuda --host 0.0.0.0 --port 5560 \
  > /tmp/gr00t_server.log 2>&1 < /dev/null & disown

grep -m1 "Server ready" <(tail -f /tmp/gr00t_server.log)
```

`setsid` matters — without it the server dies with the shell that launched it.

---

## 2. The sim, the MuGS hybrid, and the camera image

### 2a. The MuJoCo sim — our scene, on DDS domain 99 / `lo`

This is the process the WBC drives. It carries `scene_g1.xml` and opens the
MuJoCo viewer.

```bash
cd /mnt/ssd2/robosnap
PYTHONPATH=/mnt/ssd2/vla_for_simulation_testing:$PYTHONPATH \
MUJOCO_GL=glfw DISPLAY=:1 setsid nohup $VENVSIM scripts/run_mujoco_wbc_sim.py \
    --scene-dir outputs/g1_ok \
    --mjcf outputs/g1_ok/mujoco/scene_g1.xml \
    > /tmp/sim.log 2>&1 < /dev/null & disown
tail -5 /tmp/sim.log
```

**No elastic band in this sim** — the dsrl `9` / `kill -USR1` step does not
apply here. That trap belongs to gear_sonic's `run_sim_loop.py`, not to ours.

### 2b. The state relay — DDS to UDP, so the photoreal windows can follow

`.venv_sim` has `unitree_sdk2py` and no gsplat; `robosnap-sim` has gsplat and no
`unitree_sdk2py`. Nothing has both, so the pose crosses as plain UDP. Pure
observer — publishes nothing to DDS.

```bash
setsid nohup $VENVSIM scripts/wbc_state_relay.py \
    --domain 99 --interface lo --port 5601 --rate 50 \
    > /tmp/relay.log 2>&1 < /dev/null & disown
tail -3 /tmp/relay.log
```

### 2c. The MuGS hybrid window

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl setsid nohup $SIM scripts/mugs_viewer.py \
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

### 2d. The camera image — the ELP fisheye pair

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl setsid nohup $SIM scripts/elp_camera_publisher.py \
    --scene-dir outputs/g1_ok \
    --mjcf outputs/g1_ok/mujoco/scene_g1.xml \
    --ply outputs/g1_ok/gs_bg_notripod.ply \
    --port 5562 --width 640 --height 480 --fov-h 170 --face-size 640 --fps 6 \
    --mirror 5601 \
    > /tmp/elp_pub.log 2>&1 < /dev/null & disown

# wait for "PUB tcp" in /tmp/elp_pub.log, then the two windows
setsid nohup $SIM scripts/elp_camera_subscriber.py --port 5562 \
    --side both --topmost --pos-left 20 60 --pos-right 700 60 \
    > /tmp/elp_sub.log 2>&1 < /dev/null & disown

tail -2 /tmp/elp_sub.log     # "pair synchronous, stamp delta 0.000 ms"
```

`--mirror 5601` on 2c and 2d is what makes them show the robot the WBC is
driving. Without it each renders its own physics and the robot sags while the
sim's stands.

---

## 3. The WBC — and the `]` that everything depends on

Start the container:

```bash
cd /mnt/ssd2/vla_for_simulation_testing/deploy-wbc-on-robot
./docker/run_docker.sh --deploy --root
```

**Inside the container**, with the ROS keyboard dispatcher — not the default
`raw` stdin one:

```bash
export GR00T_WBC_TMUX_SESSION=g1_deployment && \
/root/venv/bin/python \
  /root/Projects/deploy-wbc-on-robot/gr00t_wbc/control/main/teleop/run_g1_control_loop.py \
  --wbc_version gear_wbc \
  --wbc_model_path policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx \
  --wbc_policy_class GIDecoupledWholeBodyPolicy \
  --interface lo \
  --simulator None \
  --control_frequency 50 \
  --no-enable_waist \
  --with_hands \
  --no-high_elbow_pose \
  --no-enable_gravity_compensation \
  --keyboard_dispatcher_type ros
```

Then **activate the policy**. This is the `]` press, and it is the step whose
absence looks exactly like a working stack:

```bash
source /opt/ros/humble/setup.bash
ros2 topic info /keyboard_input                       # Subscription count: 1
ros2 topic pub --once /keyboard_input std_msgs/msg/String "{data: ']'}"
# confirm in the WBC log:  Use policy action: True
```

Without it `use_policy_action` stays `False`, the WBC substitutes the *current*
lower-body pose for the policy's action, and **GR00T's targets are received and
silently discarded** — 20 Hz inference, goals flowing at 40 Hz, robot standing,
and nothing ever gets picked up. Publishing to `/Gr00tKeyboardListener` instead
is a no-op: that topic is the listener's output, not its input.

With an interactive TTY you can press `]` directly and skip the ROS dispatcher.
Headless it dies with `Keyboard listener stopped: (25, 'Inappropriate ioctl for
device')`, printed once and easy to miss.

Other keys, once active: `O` deactivate, `W`/`S` forward/back, `A`/`D` strafe,
`Q`/`E` yaw, `Z`/`SPACE` stop, `1`/`2` base height, `N`/`M` gait frequency.

---

## 4. The GR00T client inference loop

**Inside the container**, alongside the WBC:

```bash
/root/venv/bin/python \
  /root/Projects/deploy-wbc-on-robot/run_Inferene_without_client_for_test.py \
  --policy-host localhost --policy-port 5560 \
  --camera-host 127.0.0.1 --camera-port 5555 \
  --lang-instruction "pick up the red apple and place it in the woven brown basket"
```

`run_inference_smooth.py` is the same loop with `SmoothedActionExecutor` in
place of the raw buffer — async prefetch, cubic-spline horizon interpolation,
EMA + velocity clamp, temporal ensembling. Use it if the arms lurch at chunk
boundaries:

```bash
/root/venv/bin/python \
  /root/Projects/deploy-wbc-on-robot/run_inference_smooth.py \
  --smooth.ema-alpha 0.30 --smooth.max-delta-rad 0.06 --smooth.chunk-blend-steps 8
```

It reads state from `STATE_TOPIC_NAME` and publishes
`target_upper_body_pose` on `CONTROL_GOAL_TOPIC`, which the WBC consumes —
which is why step 3's `]` has to have happened first.

---

## The one link that is not wired yet

Step 4 needs a **camera server on port 5555**. `scripts/run_mujoco_wbc_sim.py`
publishes no image stream, and our ELP publisher emits `elp/left` / `elp/right`
on 5562, which `sdk.create_observation_reader` does not recognise. So step 4
will start and then sit with no observation.

The good news is that it is one topic, not four. `new_clinet.py` maps four VLA
cameras because the teleop dataset had them, but **our** embodiment tag
(`unitree_g1_full_body_with_waist_height_nav_cmd`) takes a single `ego_view` —
that is exactly what `play_groot_wbc.py` sends:

```python
obs = {"video": {"ego_view": ...}, "state": {...}, "language": {...}}
```

So closing it means one bridge: subscribe `elp/left` on 5562, republish it under
the name `create_observation_reader` expects on 5555. Small, but it does not
exist yet.

**Until it does, the working path to get the checkpoint onto the arms in the
photoreal scene is the in-process harness**, which renders the ego view and
hands it straight to the policy server — no camera server, no DDS, no container:

```bash
bash scripts/run_all_g1.sh --with-groot
```

That is steps 1 and 4 fused into one process. It does not use the deployed WBC
(it runs the same two ONNX nets directly), so it is not a substitute for
validating the real controller — but it is the one that runs today.

---

## Health check, before trusting anything

```
pelvis height at reset        ~0.745        (route A measured +0.745 vs commanded 0.74)
WBC log                       Use policy action: True
fisheye subscriber            "pair synchronous, stamp delta 0.000 ms"
hybrid window                 shows MIRROR <n> frames, n climbing
```

A stack that missed the `]` looks identical to one that is working badly.

## Stopping everything

```bash
pkill -f wbc_state_relay; pkill -f elp_camera; pkill -f mugs_viewer
pkill -f run_mujoco_wbc_sim; pkill -f run_gr00t_server; pkill -f play_groot_wbc
/mnt/ssd2/vla_for_simulation_testing/deploy-wbc-on-robot/docker/kill_all_containers.sh
```

Never `pkill -f <pattern>` from a shell whose own command line contains that
pattern — it matches itself and kills your shell.
