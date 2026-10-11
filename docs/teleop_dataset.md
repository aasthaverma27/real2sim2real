# Teleoperation and dataset recording in a reconstructed scene

Once a scene is built ([scene_pipeline.md](scene_pipeline.md)), this records a
LeRobot dataset in it: a person teleoperates the simulated G1 with a Pico VR
headset, the deployed whole-body controller (WBC) moves it, and the exporter
records the robot's own three cameras, rendered photoreal from the
reconstructed room. Every episode gets a new object layout and a new lighting
look.

```
 host (this repo)                                     decoupled_wbc container (external/)
 ───────────────────────────────────────────          ─────────────────────────────────
 run_mujoco_wbc_sim.py   MuJoCo, steps the  ◄─ DDS 99 ─►  run_g1_control_loop.py   WBC
   scene_wbc.xml         sim scene                         ▲
   │ UDP: robot + objects + look seed                      │ ROS 2
   ├─► mugs_viewer.py          hybrid window               run_teleop_policy_loop.py ◄── Pico
   ├─► elp_camera_publisher.py head camera window          │
   └─► vla_camera_server.py    3 cameras, tcp://*:5555 ───► run_g1_data_exporter.py ─► LeRobot
```

Nothing here uses the reference stack's own simulator. Our sim steps the
reconstructed scene, speaks the same DDS topics as the real robot (domain 99
on `lo` rather than domain 0 on the robot network), and the controller and
teleop code run unchanged.

## Prerequisites

- A built scene, `outputs/myroom/` (see [scene_pipeline.md](scene_pipeline.md)).
- NVIDIA's [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)
  in `external/`, set up with `bash scripts/setup_external.sh`. It provides
  the whole-body controller, the Pico teleop loop and the LeRobot exporter
  (`decoupled_wbc`, run in NVIDIA's container), and the DDS bridge env the
  simulator runs in (`.venv_sim`). The setup script applies
  `patches/GR00T-WholeBodyControl_sim_teleop.patch`: the ZMQ camera client the
  exporter records from (re-subscribing when the camera server restarts),
  right-hand grasp smoothing, the controller's DDS domain set to the sim's 99,
  and the three-camera recording defaults.
- The container image, pulled once:
  `cd external/GR00T-WholeBodyControl && ./decoupled_wbc/docker/run_docker.sh --install --root`.
- A Pico 4 / 4 Ultra headset running the XRoboToolkit app, on the same network
  as this machine.
- About 9 GB of free VRAM for the stack with `--vla`.

## 1. The stack (host)

```bash
scripts/realsim/start.sh --vla --visual-dr --scene outputs/myroom
```

This starts the MuJoCo sim, the MuGS hybrid window, the head camera window,
the VLA camera server and the wrist camera windows. Wait for `vla cameras . up`.
The robot falls under gravity until a controller attaches; that is expected.

| flag | effect |
|---|---|
| `--vla` | serve the three cameras the exporter records, on `tcp://*:5555` |
| `--visual-dr [CONFIG]` | a new lighting / material look on every reset (default config `configs/visual_randomization.yaml`) |
| `--fixed-layout` | do not reshuffle the objects on reset |
| `--layout-seed N` | reproducible layouts |
| `--hold` | hold the initial pose instead of falling |
| `--paused` | start with physics stopped, to look at a scene |
| `--builtin` | run a bundled balance controller in-process instead of the deployed WBC (no container needed; no teleop) |
| `--stop` | stop everything this script started |

The hybrid window frames itself automatically. To fix a view, save it as
`outputs/myroom/realsim_view.json`:

```json
{ "look_at": [0.02, 0.83, 0.66], "distance": 1.97, "azimuth": -93.65,
  "elevation": 25.14, "pos": [0, 30], "fovy": 55.21,
  "exclude_geoms": "floor,ground,room,obj0_round_wooden_table" }
```

`exclude_geoms` lists meshes the renderers should not draw because the splat
already shows them (a static table keeps its collision, so objects still rest
on it).

## 2. The whole-body controller (container)

Open the container (it starts an interactive shell; leave it open):

```bash
cd external/GR00T-WholeBodyControl && ./decoupled_wbc/docker/run_docker.sh --root
```

Then, from another terminal (`start.sh` also prints this command):

```bash
docker exec -it -w /root/Projects/GR00T-WholeBodyControl \
    -e PYTHONPATH=/root/Projects/GR00T-WholeBodyControl decoupled_wbc-bash-root bash -c '
  source /root/venv/bin/activate; source /opt/ros/humble/setup.bash
  python decoupled_wbc/control/main/teleop/run_g1_control_loop.py \
    --wbc_version gear_wbc \
    --wbc_model_path policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx \
    --wbc_policy_class G1DecoupledWholeBodyPolicy \
    --interface lo --simulator None --control_frequency 50 \
    --no-enable_waist --with_hands --no-high_elbow_pose --no-enable_gravity_compensation'
```


Press `]` in this terminal and confirm `Use policy action: True`. Keys typed
here are republished on the ROS topic `/Gr00tKeyboardListener`, which is how
the teleop loop and the exporter hear them. Three details fail silently:

- **`PYTHONPATH`**: makes the mounted, patched checkout win over the copy baked
  into the image, whose config says `DOMAIN_ID: 0` (the real robot). Without
  it the controller starts cleanly and talks to nothing, and the sim keeps
  printing `WAITING FOR WBC`.
- **`-it`**: the keyboard dispatcher reads stdin. Without a TTY, `]` never lands.
- **the two `source` lines**: `docker exec` skips `.bashrc`, so without them
  the result is `ModuleNotFoundError: rclpy`. Append to `PYTHONPATH`; never
  replace it.

## 3. Teleoperation (container)

In the headset app, enter **this machine's** IP address as seen from the
headset (`ip route get <headset-ip>` prints it as `src`). The container uses
`--network=host`, so no ports need mapping. Then, in a second container shell:

```bash
docker exec -it decoupled_wbc-bash-root bash
source /root/venv/bin/activate; source /opt/ros/humble/setup.bash
cd /root/Projects/GR00T-WholeBodyControl
PYTHONPATH=/root/Projects/GR00T-WholeBodyControl:$PYTHONPATH \
python decoupled_wbc/control/main/teleop/run_teleop_policy_loop.py \
  --body_control_device pico --hand_control_device pico \
  --body_streamer_ip <HEADSET_IP> --body_streamer_keyword knee \
  --no-enable_waist --no-high_elbow_pose --no-enable_visualization \
  --enable_real_device --upper-body-joint-speed 50 \
  --teleop-frequency 50 --control_frequency 50 --no-binary-hand-ik
```

| what | Pico | keyboard (WBC terminal) | confirms with |
|---|---|---|---|
| lower body on | left menu + left trigger | `]` | `Use policy action: True` |
| arms on | left menu + right trigger | `l` | `Starting teleop policy` → `Teleop policy calibrated` |
| reset arm calibration | | `k` | |

Until the arms are activated the teleop loop publishes a frozen neutral pose:
data flows, nothing moves. Activation calibrates against the controllers'
current pose, so hold a neutral posture when you trigger it.

To check the headset link without the robot, run the streamer test in the
container: `python decoupled_wbc/control/teleop/streamers/pico_streamer.py`.
Identity matrices mean the service is up but no headset is streaming; live
poses that change as you move mean it works.

## 4. Recording (container)

Gate first. Both must print something:

```bash
ros2 service list | grep robot_config                       # the WBC is up
python decoupled_wbc/scripts/zmq_probe.py --connect tcp://localhost:5555 --seconds 3
#   topics seen: stereo/right, hand_left, hand_right
```

Then, in a third container shell:

```bash
PYTHONPATH=/root/Projects/GR00T-WholeBodyControl:$PYTHONPATH \
python decoupled_wbc/control/main/teleop/run_g1_data_exporter.py \
  --data_collection_frequency 20 --root_output_dir outputs \
  --lower_body_policy gear_wbc \
  --wbc_model_path "policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx" \
  --camera_host localhost --camera_port 5555
```

Answer `n` to create a new dataset. It is named `<timestamp>-G1-sim`. Answering
`y` with a name that does not exist yet makes LeRobot try HuggingFace and fail
with a 401. `--data_collection_frequency 20` must match the camera server's
20 fps: a faster exporter trips its own timing monitor.

| key (WBC terminal) | effect |
|---|---|
| `c` | idle → recording → save → idle (press it to start, again to stop and save) |
| `x` | discard the episode being recorded |

The Pico `A` / `B` buttons map to the same actions only if the controller
build routes them onto `/Gr00tKeyboardListener`. If a press does not change
the episode state, use the keys. Datasets land in
`external/GR00T-WholeBodyControl/outputs/` (`$RS_DEPLOY_ROOT/outputs/`).

Between episodes, press **backspace in the MuJoCo sim window**. That resets
the sim, places the objects at a new reachable layout and, with `--visual-dr`,
applies a new look.

**Start in this order:** stack → WBC → teleop → exporter. The exporter is the
one component that cannot recover from starting early.

## Randomization

**Object layout** (`scripts/realsim/randomize_objects.py`, on by default). On
every reset the movable objects are placed on the table top, inset by their
own footprint, inside the robot's reach (0.20–0.42 m from the base, −80° to
+45° about its forward axis, the envelope of our right-arm demonstrations), and
kept apart so the apple never starts inside the basket. `--fixed-layout`
turns it off and `--layout-seed N` makes it reproducible.

**Visual** (`scripts/realsim/visual_randomizer.py`, with `--visual-dr`). Per
reset: key-light direction and intensity (so shadows move), a spot light aimed
at the table, light colour from warm to cool, headlight fill, table / apple /
basket brightness, tint and gloss, and a colour jitter on the splat background,
whose lighting is otherwise baked in. The apple stays red and the basket brown,
because the task instruction names them. Robot colours are not randomized.
Ranges live in `configs/visual_randomization.yaml`. Physics is never touched.

The sim draws one seed per reset and broadcasts it next to the object poses.
Every renderer turns the same seed into the same look, so the windows and the
recorded cameras always agree, and any episode's look can be re-derived from
its seed (seed 0 is the nominal scene). Preview the looks before a session:

```bash
scripts/realsim/preview_visual_dr.py --seeds 0-15 --scene-dir outputs/myroom
#   -> contact sheet image
```

## 5. Finalize the dataset (host)

The exporter's `x` does not delete an episode: it saves it and lists its index
under `discarded_episode_indices` in `meta/info.json`. To build a clean dataset,
dropping discarded episodes, renumbering and merging sessions:

```bash
scripts/realsim/finalize_dataset.sh my-apple-pnp-G1-sim \
    2026-09-29-11-20-00-G1-sim 2026-09-29-12-05-31-G1-sim
```

The names are folders under `$RS_DEPLOY_ROOT/outputs/`.
It runs `gear_sonic`'s `process_dataset.py` inside the WBC container, the only
environment with its dependencies, and leaves the source sessions untouched.

## What gets recorded

Each episode, at 20 Hz:

| feature | shape | from |
|---|---|---|
| `observation.images.ego_view` | 480 × 640 video | `stereo/right` → head camera |
| `observation.images.ego_left` | 480 × 640 video | `hand_left` → left wrist camera |
| `observation.images.ego_right` | 480 × 640 video | `hand_right` → right wrist camera |
| `observation.state` | 43 | joint positions (legs 0–12, waist 12–15, left arm 15–22, left hand 22–29, right arm 29–36, right hand 36–43) |
| `action` | 43 | joint targets |
| `observation.eef_state`, `action.eef` | 14 | left / right wrist position + quaternion, relative to the base |

All three views are rendered from one simulator state and share a frame id and
timestamp, so every recorded step is internally consistent.

## The camera server

The exporter never touches the sim: it reads robot state from ROS and images
from a ZMQ camera server. Two camera protocols exist in the controller stack, and
`scripts/realsim/vla_camera_server.py` speaks both from one port:

| consumer | wire format | names |
|---|---|---|
| `decoupled_wbc` exporter and VLA client (`VLAImageClient`, from our patch) | 3-part `[topic, meta json, jpeg]` | `stereo/right`, `hand_left`, `hand_right` |
| gear_sonic exporter (`ComposedCameraClientSensor`) | 1-part msgpack, base64 JPEGs | `ego_view`, `ego_left`, `ego_right` |

`start.sh` uses the multipart protocol by default (`VLA_PROTOCOL=multipart`).
A SUB socket with a topic filter silently drops single-part msgpack.

### Ports

| port | what |
|---|---|
| DDS domain 99, interface `lo` | sim ↔ WBC (`rt/lowcmd`, `rt/lowstate`, `rt/dex3/*`) |
| UDP 5601 / 5602 / 5603 | robot + object state and look seed → hybrid window / head camera / VLA camera server |
| TCP 5562 | head camera stream (`elp/left`, `elp/right`) |
| TCP 5555 | VLA cameras (`stereo/right`, `hand_left`, `hand_right`) |

Set `EXTRA_MIRROR_PORTS=5604` to give a further process its own copy of the
object stream.

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| sim prints `WAITING FOR WBC` forever | controller on DDS domain 0 | the `PYTHONPATH` pin in step 2, and the patch from `setup_external.sh` |
| `ModuleNotFoundError: rclpy` | `docker exec` skipped `.bashrc`, or `PYTHONPATH` was replaced instead of appended | the two `source` lines; `PYTHONPATH=...:$PYTHONPATH` |
| `]` does nothing | no TTY | `docker exec -it` |
| exporter: `image False` / `NO ZMQ frames` | no camera server, or it restarted under the exporter | start the stack with `--vla`; the patched client re-subscribes after 5 s of silence |
| `OSError: ... crc_amd64.so: invalid ELF header` | a Git LFS file was not fetched | `bash scripts/setup_external.sh` (fetches the needed LFS files) |
| exporter hangs before `Recording to ...` | a second exporter is running (both are `/data_exporter`) | stop the other one |
| `Address already in use tcp://*:5562` | leftovers from a crashed run | `scripts/realsim/start.sh --stop`, then start again |
| objects frozen in the windows while the sim's move | stack started without the sim's object stream | always start through `start.sh` |
| sim looks slow | check the `rtf=` value on the sim's status line | expect about 1.0×; close other GPU jobs |
| Pico poses stay identity | headset not paired or wrong host IP in the app | `ip route get <headset-ip>`, re-enter the `src` address |

Logs for every host process are under `/tmp/realsim/`.
