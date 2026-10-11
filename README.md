# real2sim2real

**A photoreal, metric digital twin of a real room, built from a phone video, in
which a Unitree G1 humanoid is teleoperated through its real whole-body
controller to record VLA training data for zero-shot deployment.**

The aim is to train robot policies entirely in simulation and run them on the
real robot without real-world data. That only works if the simulator removes
both gaps a policy can see:

- **Visual gap.** The room is a 3D Gaussian splat trained on the real room. The
  robot and the objects are MuJoCo meshes composited into it. The robot's three
  cameras are rendered with checkerboard-calibrated intrinsics and CAD-measured
  mounts, through the same crop-and-resize chain as the real camera server.
- **Dynamics and control gap.** The scene is in true metric scale. The objects
  have collision hulls and physical properties. The robot is driven by the
  **same deployed whole-body controller** (NVIDIA GR00T-WBC) over the same DDS
  interface as the real G1. Only the DDS domain changes between sim and robot.

Demonstrations are recorded by VR teleoperation inside this twin, with object
layout and lighting randomized every episode, and exported as a LeRobot dataset
ready for VLA fine-tuning (e.g. GR00T N1.x).

<p align="center"><img src="assets/visual_dr_preview.jpg" width="900"></p>
<p align="center"><sub>The robot's head camera in the reconstructed room: the splat background, the mesh table, apple and basket, and the robot's own hands, under 16 visual-randomization seeds.</sub></p>

```
 phone video ─► COLMAP ─► 3DGS (objects painted out) ─► auto-align (floor, lens, scale)
                                                              │
 one photo ─► SAM 3 masks ─► SAM-3D meshes ─► collision + physics ─► metric MuJoCo scene
                                                              │
               G1 + calibrated camera rig ─────────────► scene_g1.xml (render) / scene_wbc.xml (sim)
                                                              │
   Pico VR ─► teleop ─► GR00T-WBC ◄── DDS 99 ──► MuJoCo sim ──┼─► MuGS hybrid render (3 robot cameras)
                                                              └─► LeRobot exporter ─► dataset
```

## Contents

1. [Install](#1-install)
2. [Capture a room](#2-capture-a-room)
3. [Build the scene](#3-build-the-scene)
4. [Calibrate the robot's cameras](#4-calibrate-the-robots-cameras)
5. [Run the photoreal simulation](#5-run-the-photoreal-simulation)
6. [Teleoperate and record a dataset](#6-teleoperate-and-record-a-dataset)
7. [How it works](#7-how-it-works)
8. [Repository layout](#8-repository-layout)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Install

**Requirements.** Ubuntu 22.04 or later, an NVIDIA GPU with ≥ 16 GB of VRAM
(24 GB+ recommended; building a scene peaks around 12 GB, the full teleop stack
around 9 GB), a driver for CUDA 12.8, conda, git, ffmpeg. Teleoperation also
needs docker, git-lfs and a Pico 4 / 4 Ultra headset.

```bash
git clone https://github.com/MethdAI-Modinoid/real2sim2real.git
cd real2sim2real

huggingface-cli login        # or export HF_TOKEN; SAM 3 / SAM-3D-Objects weights are gated - accept their terms on HF first
bash scripts/install_auto_pipeline.sh -y --download-checkpoints   # envs + model weights + configs/paths.env
bash scripts/fetch_g1_meshes.sh                                   # stock Unitree G1 meshes (not committed)
bash scripts/doctor.sh                                            # checks everything; ends in "ready"
```

The installer fetches the pinned model sources (SAM 3, SAM-3D-Objects, VGGT),
applies `patches/sam-3d-objects_gsplat_sm120.patch` (a gsplat renderer for GPUs
without the inria rasterizer, e.g. RTX 50xx), creates the environments, and
writes `configs/paths.env`. Every script reads that file, so nothing has to be
sourced by hand.

| environment | holds | used for |
|---|---|---|
| `robosnap-sim` | mujoco, gsplat, MuGS, OpenCV, zmq, onnxruntime | scene export, rendering, simulation viewers |
| `robosnap-lyra` | pycolmap, gsplat training | COLMAP, 3DGS training, splat alignment |
| `robosnap-asset` | SAM-3D-Objects, VGGT, trimesh | object meshes |
| `robosnap-sam3` | SAM 3 | segmentation |
| `.venv_sim` (built by `setup_external.sh`) | unitree_sdk2py, gear_sonic | the DDS simulator |

They pin conflicting CUDA and torch builds, which is why they are separate.

<details>
<summary><b>The whole-body controller, teleop and recording stack</b> (sections 5–6)</summary>

```bash
bash scripts/setup_external.sh            # NVIDIA GR00T-WholeBodyControl + our patch + .venv_sim
cd external/GR00T-WholeBodyControl && ./decoupled_wbc/docker/run_docker.sh --install --root   # container image
```

Teleoperation and recording run on NVIDIA's public
[GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)
(Apache-2.0; model weights under the NVIDIA Open Model License): the deployed
whole-body controller, Pico teleop, the LeRobot exporter and the MuJoCo ↔ DDS
bridge. It is the repository's only submodule, pinned to a tested commit, and
runs in NVIDIA's prebuilt container. Our changes to it are one patch,
`patches/GR00T-WholeBodyControl_sim_teleop.patch`: the camera client the
exporter records from, right-hand grasp smoothing, the sim's DDS domain (99)
and the three-camera recording defaults. Details:
[docs/external_dependencies.md](docs/external_dependencies.md).

Sections 2–5 work without it, with a balance controller run in-process
(`start.sh --builtin`, using the same ONNX policies).
</details>

---

## 2. Capture a room

**Take the objects off the table, film the room, put them back, take one photo.**

1. Move the robot out of the room. Clear the objects you want to manipulate
   off the table; leave the table and everything that does not move.
2. Film one continuous take of 40–60 s with a phone: walk a slow loop, keep
   the room in view, **vary the camera height** (crouch, stand, overhead),
   lock exposure and white balance, and keep people out of the shot.
3. Put the objects back where you want them and take **one photo**, roughly
   from where the video started. Every object must be visible in it.
4. Measure the table height (and one more length, e.g. the basket, as a check).

An object filmed in the room cannot be cleanly removed from a splat afterwards.
If you cannot re-shoot, the pipeline can paint objects out of the frames
(`--mask-objects`), at the cost of a soft patch where they stood. Why each
removal method leaves a mark, and what makes a clean background:
[docs/scene_pipeline.md §0](docs/scene_pipeline.md#0-capture).

## 3. Build the scene

One command. Each stage is skipped when its output exists, so a re-run resumes.

```bash
bash scripts/build_sim_scene.sh \
    --scene-dir outputs/myroom \
    --room-video room_no_objects.mp4 \
    --object-image objects_on_table.jpg --object-free-video \
    --objects "the round wooden table,the red apple,the woven brown basket" \
    --table-body obj0_round_wooden_table --table-height 0.75 \
    --check-body obj2_woven_brown_basket --check-size 0.245 0.19 \
    --robot-at 0.0 0.42 --robot-yaw 1.54
```

| stage | does | output |
|---|---|---|
| `capture` | bakes the phone's rotation flag, extracts the first frame | `inputs/room.mp4`, `inputs/frame0.png` |
| `objects` | SAM 3 masks → SAM-3D meshes → collision hulls | `fully_refined_foreground.glb` |
| `properties` | mass, friction, restitution by material lookup | `physical_properties.json` |
| `export` | MuJoCo scene | `mujoco/scene.xml` |
| `rescale` | one measured length → metres (the second length checks it) | `mujoco/scene_scaled.xml` |
| `colmap` | camera poses for the whole video, one model | `roommesh/colmap/` |
| `masks`, `inpaint` | only with `--mask-objects`: paint the objects out of every frame | `roommesh/images_clean/` |
| `splat` | 3D Gaussian Splatting, 30k iterations (~13 min, ~7 GB) | `gs_wide.ply` |
| `align` | gravity from the splat's floor plane, lens from COLMAP, scale from the table height | `gs_bg.ply` |
| `reseat` | moves each object onto the splat where the photo shows it | updated `scene_scaled.xml` |
| `prune` | only in the fallback mode: cuts objects out of the splat | `gs_bg_clean.ply` |
| `robot` | attaches the G1 with the calibrated camera rig | `mujoco/scene_g1.xml` (rendered) |
| `wbc` | robot-first re-merge for the controller | `mujoco/wbc/scene_wbc.xml` (simulated) |
| `verify` | checks every fault that once failed silently | report; exit code = failures |

`--dry-run` prints every command. `--from STAGE` resumes, `--only STAGE` runs
one, `--force` redoes. `--robot-at X Y` / `--robot-yaw RAD` are where the robot
stands; choose them in `python -m mujoco.viewer --mjcf outputs/myroom/mujoco/scene_scaled.xml`.
If the GPU is busy, `bash scripts/wait_for_vram.sh 9000 -- bash scripts/build_sim_scene.sh ...`
starts the build when there is room.

Every stage, flag and design decision, with a worked example:
**[docs/scene_pipeline.md](docs/scene_pipeline.md)**.

## 4. Calibrate the robot's cameras

Skip this if you use our G1: its measured values are already in
`assets/robots/g1/g1_cad.xml` and `calib/`. For your own robot, calibrate each
camera's **published** stream (the frame the policy actually sees) with a
checkerboard:

```bash
sudo modprobe v4l2loopback video_nr=9 card_label=robot_cam exclusive_caps=1
scripts/realsim/zmq_to_v4l2.py --topic stereo/right --device /dev/video9    # leave running
scripts/realsim/run_opencv_calib.sh stereo/right 9 6 25.0                   # board: inner corners, mm
scripts/realsim/read_opencv_calib.py calib/stereo_right_opencv.xml \
    --camera head --topic stereo/right --write calib/stereo_right.yaml      # -> MJCF fovy
```

| camera | fovy (MJCF) | spread across runs |
|---|---|---|
| head stereo (`stereo/left`, `stereo/right`) | 53.35° | 0.40° / 0.00° |
| left wrist (`hand_left`) | 63.34° | 5.14° (re-measure) |
| right wrist (`hand_right`) | 63.34° | 0.02° |

Camera positions come from the CAD of the camera mounts
(`assets/robots/g1/meshes/cad/`). The head pitch (56.7°) was measured with
solvePnP. Procedure, the traps that silently reject views, and which error
numbers to trust: **[docs/camera_calibration.md](docs/camera_calibration.md)**.

## 5. Run the photoreal simulation

```bash
scripts/realsim/start.sh --scene outputs/myroom --builtin          # self-contained
scripts/realsim/start.sh --scene outputs/myroom                    # waits for the deployed WBC
scripts/realsim/start.sh --stop
```

This opens the MuJoCo sim, the MuGS hybrid window (the robot and objects
composited into the splat) and the robot's head-camera stream. With `--builtin`
a balance controller runs in-process. Without it, the sim waits on DDS domain
99 for the deployed whole-body controller, exactly as the real robot does on
domain 0; `start.sh` prints the container command. `--paused` and `--hold` are
for inspecting a scene. Logs are written to `/tmp/realsim/`.

## 6. Teleoperate and record a dataset

```bash
scripts/realsim/start.sh --scene outputs/myroom --vla --visual-dr   # host: sim + 3 robot cameras on :5555
#   container: GR00T-WBC (press ]), Pico teleop, LeRobot exporter (keys c / x)
scripts/realsim/finalize_dataset.sh my-dataset <session> [<session> ...]
```

Each reset (backspace in the sim window) puts the objects at a new reachable
layout and, with `--visual-dr`, gives the scene a new lighting and material
look. The look comes from one broadcast seed, so every camera agrees and any
episode can be re-rendered. The dataset records the head and both wrist
cameras (480 × 640), 43-D joint state and action, and wrist poses at 20 Hz.

The container commands, Pico setup, recording keys, randomization ranges,
ports and troubleshooting: **[docs/teleop_dataset.md](docs/teleop_dataset.md)**.

---

## 7. How it works

**One splat, one take.** The background is a single 3DGS reconstruction of one
continuous capture, trained against the frames with COLMAP poses. Two captures
stitched together leave a blurred seam where they meet. A feed-forward
reconstruction averages views that disagree instead of reconciling them.

**Alignment without fitting.** The object photo is registered in the same
COLMAP model as the video, so its camera is an exact reference shared by both
reconstructions. Everything else is read off the splat rather than the
single-view object reconstruction: gravity from a RANSAC floor plane over tens
of thousands of points, the lens from bundle adjustment over 450 views, and
scale from the splat's floor-to-tabletop distance against the measured table
height. Objects are then re-seated onto the splat by ray-casting their masks.

**Two scenes.** The renderer needs the robot attached *into* the room,
namespaced `g1_`. The deployed controller needs the robot's own joint names
with its floating base at joint 0. So `build_wbc_scene.py` merges the other
way round, with the robot as parent and the room re-originated onto its stance.
`scene_wbc.frame.json` records that shift and every renderer undoes it.

**State flow.** The sim (`run_mujoco_wbc_sim.py`) steps `scene_wbc.xml`,
bridges DDS to the controller, and streams the robot's joints, every free
object's pose and the current visual-randomization seed over UDP. Each renderer
(the hybrid window, the head camera, the VLA camera server) loads `scene_g1.xml`
plus the splat and mirrors that state. They run in a different Python
environment from the DDS bridge, because no single environment has both
`unitree_sdk2py` and `gsplat`.

**Compositing.** Following Re3Sim, static support surfaces are not drawn as
meshes: listed in `realsim_view.json → exclude_geoms`, they keep their
collision and the splat shows the real surface. MuGS (vendored in
`third_party/MuGS`, our changes in `patches/` and
`third_party/MuGS/FORK_NOTES.md`) composites the MuJoCo foreground over the
gsplat background.

**Camera parity.** The sim renders at each stream's published resolution
(head 768 × 432, wrists 1280 × 720) with the measured fovy. The camera server
then applies the real client's anisotropic resize to 640 × 480. The two
protocols the controller stack uses (3-part multipart and msgpack) are both served
from one port.

What was measured and what failed along the way (compositing direction,
splat coverage of the robot's view, why five automated yaw objectives
disagreed, the fisheye projection): [docs/real2sim_experiments.md](docs/real2sim_experiments.md).

## 8. Repository layout

```
scripts/
  build_sim_scene.sh          one-command scene build (section 3)
  doctor.sh                   environment check
  realsim/start.sh            the simulation stack (sections 5, 6)
  realsim/                    WBC scene merge, verification, camera server,
                              randomizers, calibration tools, dataset finalizer
  *.py                        the individual stages (colmap, train_gs, align, reseat, ...)
configs/
  paths.env.example           machine-specific locations (installer writes paths.env)
  robots/g1_cad.yaml          robot descriptor: joints, gains, cameras, WBC
  visual_randomization.yaml   randomization ranges
assets/robots/g1/             G1 MJCFs, camera-rig CAD meshes
calib/                        measured intrinsics of our robot's cameras
robosnap/                     object reconstruction package (from RoboSnap)
third_party/MuGS/             vendored Gaussian-splat sensor
patches/                      our changes to third-party code
docs/                         detailed documentation
external/                     submodule: NVIDIA GR00T-WholeBodyControl (controller, teleop, exporter)
```

Generated scenes go to `outputs/` and model weights to `checkpoints/`; both
are gitignored.

## 9. Troubleshooting

| symptom | cause / fix |
|---|---|
| anything fails on start | `bash scripts/doctor.sh` names the missing piece and the command that fixes it |
| `Ninja is required to load C++ extensions` | gsplat's JIT needs the sim env on `PATH`; run through the provided scripts, which source `scripts/_bootstrap.sh` |
| `[pipeline] WARNING: the capture did not reconstruct as one room` | `--matcher exhaustive --force --only colmap` |
| `align` stops with exit code 3 | the capture could not be aligned automatically; the message lists what to check |
| two tables (photoreal and solid) a hand-span apart | `reseat` skipped: it needs `--table-body` and `--table-height` |
| robot model edits do not show up | built scenes inline the robot; rebuild with `--only robot --force`, then `--only wbc --force` |
| `--force` reset a hand-tuned object scale | `--force` regenerates every stage it covers; tune after the last forced rebuild |
| sim prints `WAITING FOR WBC` forever | the controller is on DDS domain 0; see [docs/teleop_dataset.md](docs/teleop_dataset.md#troubleshooting) |
| `Address already in use` | leftovers from a crashed run: `scripts/realsim/start.sh --stop` |
| robot's view mostly black | the splat does not cover the robot's viewpoint: film from the robot's eye height too |
| `EGLError` at interpreter exit | teardown noise, harmless |

## Documentation

| document | covers |
|---|---|
| [docs/scene_pipeline.md](docs/scene_pipeline.md) | capture rules, every build stage, alignment and reseating, verification, worked example |
| [docs/camera_calibration.md](docs/camera_calibration.md) | checkerboard intrinsics, CAD extrinsics, our measured values |
| [docs/teleop_dataset.md](docs/teleop_dataset.md) | Pico teleop, the controller container, recording, randomization, finalizing |
| [docs/mugs_hybrid_render.md](docs/mugs_hybrid_render.md) | the MuGS bridge and its performance |
| [docs/real2sim_experiments.md](docs/real2sim_experiments.md) | findings F1–F15: what was measured and what failed |
| [docs/external_dependencies.md](docs/external_dependencies.md) | NVIDIA GR00T-WholeBodyControl: what we use, our patch, licences |
| [docs/automatic_pipeline_setup.md](docs/automatic_pipeline_setup.md) | installer details and model checkpoints |
| [assets/robots/g1/README.md](assets/robots/g1/README.md) | the robot models and camera rig |

## Credits

- **[RoboSnap](https://robosnap.github.io)**: the object-reconstruction
  pipeline (`robosnap/`, the GUI and the automatic layered reconstruction) that
  this project builds on.
- **[MuGS](https://github.com/Renforce-Dynamics/MuGS)**: the MuJoCo
  Gaussian-splat sensor, vendored in `third_party/MuGS` with our patch.
- **[Re3Sim](https://xshenhan.github.io/Re3Sim/)**: the semantic-substitution
  compositing rule. The method is theirs; no code is copied.
- **SAM 3**, **SAM-3D-Objects**, **VGGT**, **COLMAP**, **gsplat**: segmentation,
  object meshes, depth, camera poses and splat training, each under its own
  licence.
- **[NVIDIA GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)**:
  the deployed whole-body controller and its policies, Pico teleop, the LeRobot
  exporter and the DDS bridge, used as a submodule with our patch.
- **Unitree**: the G1 robot model and SDK.

Licensed under Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). This
applies to first-party code only. Third-party components, submodules and model
weights keep their own licences.
