# G1 apple scene — handover

Everything done on `outputs/g1_ok/`, why it was done that way, and every command
needed to bring it back up. Written to survive a context loss: if you are a new
session, read this file and `docs/real2sim_experiments.md` (findings F1–F14)
before touching the alignment or the background.

Companion documents:

| file | holds |
|---|---|
| `RUNBOOK_g1.md` | how the scene was *built*, stage by stage, from the phone video |
| `RUNBOOK_now.md` | the short copy-paste command list |
| `docs/real2sim_experiments.md` | F1–F14: what was measured, what failed, why |
| `docs/groot_wbc_photoreal.md` | the GR00T + WBC harness design |
| `docs/g1_full_stack.md` | the four-process DDS stack (policy server -> sim -> WBC container -> client) with the photoreal windows mirrored onto it |
| this file | current state, the whole narrative, and the live commands |

---

## 0. Standing constraint

**GPU must stay under 31 GB VRAM.** If a job pushes past it, stop the job, wait
an hour, check again, resume only when the card is free. The RTX 5090 here has
32,607 MiB total, so 31 GB is nearly the whole card — the real rule is *never
run two heavy stages at once*. SAM-3D peaks at ~17 GB and Lyra at ~18 GB;
together they exceed the budget and one of them dies.

Check before launching anything:

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
```

Idle is ~500 MiB. The live stack (sim + hybrid + fisheye) sits at ~3.2 GB.

---

## 1. Environment

Two interpreters, and they are **not** interchangeable:

```bash
cd /mnt/ssd2/robosnap
SIM=/mnt/ssd2/conda/envs/robosnap-sim/bin/python    # mujoco, MuGS, gsplat, open3d, cv2, zmq, onnxruntime
LYRA=/mnt/ssd2/conda/envs/robosnap-lyra/bin/python  # pycolmap, gsplat training/eval
export PATH=/mnt/ssd2/conda/envs/robosnap-sim/bin:$PATH
export DISPLAY=:1
```

The `PATH` line is not optional for anything touching gsplat: it JIT-compiles
CUDA kernels on first use and needs `ninja` from that env, otherwise it dies
with `RuntimeError: Ninja is required to load C++ extensions`.

Long jobs must be detached, or they die when a tool call times out:

```bash
setsid nohup <command> > <logfile> 2>&1 < /dev/null & disown
```

Plain `nohup … &` is not enough — that was learned the hard way when a
TextureMesh run died silently 40 minutes before anyone noticed.

**Checking whether something is actually running.** Do *not* grep a process
list inline: the grep matches its own command line and reports everything as
alive. This produced repeated false "it's still running" reports. Snapshot
first:

```bash
ps -eo pid,etime,cmd > /tmp/ps.txt
grep -E "play_groot_wbc|elp_camera|mujoco|train_gs" /tmp/ps.txt | grep -v grep
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader
```

---

## 2. What the scene is now

```
outputs/g1_ok/mujoco/scene_g1.xml      the scene: room-scale MJCF, 3 objects, G1
outputs/g1_ok/gs_bg_notripod.ply       the background to render — USE THIS ONE
outputs/g1_ok/gs_bg_clean.ply          same, before the tripod cut (has the tripod)
outputs/g1_ok/gs_bg.ply                same splat, unpruned, world frame
outputs/g1_ok/gs_wide.ply              the trained splat, still in the COLMAP frame
outputs/g1_ok/bg_lm.ply.keep           the original Lyra background, kept for reference
assets/robots/g1/                      the vendored G1 model (edit here, not upstream)
```

Measured state, all verified:

```
table       height 74.47 cm  (real 74.50)     diameter 84.39 cm  (real 85.00)
            mesh scale 1.088566 1.088566 1.285978  — non-uniform, see §4
apple       pos -0.1544  0.7199  0.7811       seated, no penetration at rest
basket      pos -0.0029  0.7031  0.7918       seated, no penetration at rest
pelvis      pos -0.0198  1.3720  0.7923       clear of the table, feet at z=0
splat floor      z = -0.0040   (MuJoCo floor 0.0000)   4 mm out
splat tabletop   z = +0.7425   (mesh tabletop 0.7447)  2 mm out
splat transform  scale 0.3965, yaw 268.00 deg
splat contents   2,140,844 gaussians  (212,719 tabletop-pruned + 81,269 tripod)
head camera      local 0.075 0 0.385, 20 deg down, on the head's front face
cameras          g1_elp_left / g1_elp_right (fisheye pair), g1_head_pinhole,
                 cap (hand-fitted to the photograph, fovy 55.21), scene_view
```

The MJCF **never names a splat** — the background is always passed with `--ply`.
Nothing in the scene file selects it.

Backups of the scene, if something needs undoing:

```
scene_g1.xml.pre_dims             before the table dimensions were corrected
scene_g1.xml.pre_elp              before the D435i -> ELP camera swap
scene_g1.xml.pitch45              before the head tilt became 20 deg
scene_g1.xml.pre_tablefit         before the table was photo-fitted
scene_g1.xml.pre_room             before the room mesh went in
scene_g1.xml.with_room_unaligned  the room mesh, misaligned — do not use
scripts/play_groot_wbc.py.pre_g1ok   before the --camera/--fovy/--no-policy flags
```

---

## 3. Live commands

### 3.1 The MuJoCo sim

```bash
MUJOCO_GL=glfw setsid nohup $SIM -m mujoco.viewer \
    --mjcf=outputs/g1_ok/mujoco/scene_g1.xml \
    > /tmp/sim.log 2>&1 < /dev/null & disown
```

In the viewer: `V` toggles camera frustums, `[` / `]` cycles cameras,
right-drag orbits.

### 3.2 The MuGS hybrid — live viewer

This drives MuGS's own `GaussianSensor`, not numpy compositing. The camera is
pinned to the world with `--look-at/--distance/--azimuth/--elevation` so that
fitting the scene cannot move the viewpoint (it used to, which silently
changed the framing between runs):

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl setsid nohup $SIM scripts/mugs_viewer.py \
    --mjcf outputs/g1_ok/mujoco/scene_g1.xml \
    --ply outputs/g1_ok/gs_bg_notripod.ply \
    --camera scene_view --capture-camera cap --fovy 55.21 \
    --width 700 --height 933 \
    --look-at 0.0216 0.8297 0.6614 --distance 1.97 \
    --azimuth -93.6502 --elevation 25.1445 \
    --splat-geoms round_wooden_table --pos 0 30 \
    > /tmp/hybrid.log 2>&1 < /dev/null & disown
```

`--splat-geoms round_wooden_table` is Re3Sim's rule: those geoms are *painted
from the splat* instead of from the mesh, so the real table's pixels win and
the mesh table never doubles it.

### 3.3 The MuGS hybrid — one still frame

```bash
MUJOCO_GL=egl $SIM scripts/mugs_render.py --scene-dir outputs/g1_ok \
    --mjcf outputs/g1_ok/mujoco/scene_g1.xml \
    --ply outputs/g1_ok/gs_bg_notripod.ply \
    --camera cap --fovy 55.21 --width 720 --height 960 \
    --output /tmp/hybrid.png
```

### 3.4 The robot's camera — ELP fisheye, two streams over ZMQ

The real camera is an **ELP-USB3DGS1200P01-L180**: global-shutter synchronous
dual-lens, 180 deg fisheye. Publisher first, then the viewer client:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl setsid nohup $SIM scripts/elp_camera_publisher.py \
    --mjcf outputs/g1_ok/mujoco/scene_g1.xml \
    --ply outputs/g1_ok/gs_bg_notripod.ply \
    --port 5560 --width 640 --height 480 --fov-h 170 --face-size 640 --fps 8 \
    > /tmp/elp_pub.log 2>&1 < /dev/null & disown

# wait for "PUB tcp" in /tmp/elp_pub.log, then
setsid nohup $SIM scripts/elp_camera_subscriber.py \
    --port 5560 --side both --topmost --pos-left 20 60 --pos-right 700 60 \
    > /tmp/elp_sub.log 2>&1 < /dev/null & disown

tail -2 /tmp/elp_sub.log        # expect "pair synchronous, stamp delta 0.000 ms"
```

**Port 5560 collides with the GR00T policy server.** If you want both, move one:
give the publisher `--port 5561` and the subscriber `--port 5561`.

Headless check, no windows:

```bash
$SIM scripts/elp_camera_subscriber.py --port 5560 --no-window --frames 8 --save /tmp/elp
```

One fisheye still, no ZMQ at all:

```bash
MUJOCO_GL=egl $SIM scripts/fisheye_render.py \
    --mjcf outputs/g1_ok/mujoco/scene_g1.xml \
    --ply outputs/g1_ok/gs_bg_notripod.ply \
    --eye both --width 640 --height 480 --fov-h 170 --face-size 768 -o /tmp/elp.png
```

### 3.5 Head tilt

Currently **20 deg down**. Set it in *both* files or only one of them changes —
`MjSpec.attach` inlines the robot, so a built scene does not read
`assets/robots/g1/` again:

```bash
$SIM scripts/set_head_tilt.py --xml outputs/g1_ok/mujoco/scene_g1.xml --prefix g1_ --pitch 20
$SIM scripts/set_head_tilt.py --xml assets/robots/g1/g1_29dof_with_hand.xml --pitch 20
```

Add `--dry-run` to see the effect without writing. The G1 has **no neck joint** —
`head_link` is a geom on `torso_link`, so tilting the head means rewriting the
camera's `xyaxes`, not moving a joint.

---

## 4. The WBC

### 4.1 Which route to use

There are two, and only one works out of the box:

| route | script | needs | status |
|---|---|---|---|
| **in-process** | `scripts/play_groot_wbc.py` | onnxruntime + the two shipped ONNX nets | **works** — use this |
| DDS | `scripts/run_mujoco_wbc_sim.py` | `unitree_sdk2py` + `gear_sonic`, DDS domain 99 on `lo` | **blocked** — neither package is installed in any conda env here |

The in-process route is also the only arrangement MuGS supports:
`sensor.render(model, data, camera)` has no step function and no notion of
another simulator, so it draws whatever `mj_data` its own process holds.
AndroidTwin's `at-eval` does exactly the same thing.

The WBC nets live at `/mnt/ssd2/simple_wbc_code/models/`:
`GR00T-WholeBodyControl-Balance.onnx` and `…-Walk.onnx`. They run on CPU.

### 4.2 WBC alone — the robot stands, no policy server

This is almost certainly what you want first. Arms and hands hold the default
stance, the navigation command is zero, and the **balance** net holds the robot
up. No GR00T server, no checkpoint, nothing else to launch:

```bash
cd /mnt/ssd2/robosnap
DISPLAY=:1 bash scripts/run_play_groot_wbc.sh \
    --scene-dir outputs/g1_ok \
    --mjcf outputs/g1_ok/mujoco/scene_g1.xml \
    --ply outputs/g1_ok/gs_bg_notripod.ply \
    --camera cap --fovy 55.21 --width 540 --height 720 \
    --no-policy --vram-cap-gb 22
```

ESC quits. The overlay shows `WBC only`, the nav command, the height command
and the live pelvis height, so you can see at a glance whether it is standing
or sagging.

Headless, recording instead of displaying:

```bash
bash scripts/run_play_groot_wbc.sh \
    --scene-dir outputs/g1_ok --mjcf outputs/g1_ok/mujoco/scene_g1.xml \
    --ply outputs/g1_ok/gs_bg_notripod.ply \
    --camera cap --fovy 55.21 --width 540 --height 720 \
    --no-policy --no-window --seconds 90 --record /tmp/wbc_only.mp4 \
    --vram-cap-gb 22
```

`--seconds` is **wall-clock, not sim time.** The MuGS composite dominates the
loop, so a run is bounded by rendering, not by physics. Use `--mode mujoco_only`
to strip the splat out and get a fast physics-only check:

```bash
bash scripts/run_play_groot_wbc.sh \
    --scene-dir outputs/g1_ok --mjcf outputs/g1_ok/mujoco/scene_g1.xml \
    --ply outputs/g1_ok/gs_bg_notripod.ply --camera cap --fovy 55.21 \
    --no-policy --no-window --seconds 30 --mode mujoco_only --vram-cap-gb 22
```

### 4.3 WBC + the GR00T checkpoint

The checkpoint's action space is

```
left_arm 7, right_arm 7, left_hand 7, right_hand 7, waist 3,
base_height_command 1, navigate_command 3
```

with **no leg actions** — it steers the base with a velocity command and the
Balance/Walk nets turn that into leg targets. That is why the WBC is not
optional here. The loop is

```
ego camera -> GR00T :5560 -> arms + hands            -> joint targets
                          \-> navigate_command + base_height
                                 -> WBC Balance/Walk -> legs + waist
                                 -> PD torques -> mj_step
                                 -> GaussianSensor.render() -> window
```

**Step 1 — the policy server** (the only other process needed):

```bash
cd /mnt/ssd2/Isaac-GR00T
export NO_ALBUMENTATIONS_UPDATE=1 HF_HOME=/mnt/ssd2/huggingface_cache \
       PYTHONPATH=/mnt/ssd2/Isaac-GR00T
setsid nohup /mnt/ssd2/dsrl/.venv/bin/python -u -m gr00t.eval.run_gr00t_server \
  --model_path /mnt/ssd2/Isaac-GR00T/checkpoints/red_green_blue_cube_sorting_in_sim/checkpoint-22000 \
  --embodiment_tag unitree_g1_full_body_with_waist_height_nav_cmd \
  --device cuda --host 0.0.0.0 --port 5560 \
  > /tmp/gr00t_server.log 2>&1 < /dev/null & disown

grep -m1 "Server ready" <(tail -f /tmp/gr00t_server.log)   # ~30 s, 7.8 GB VRAM
```

**Step 2 — the harness.** Note `--ego-camera`: this scene has no `g1_ego_view`,
which is the script's default. Use the pinhole head camera:

```bash
cd /mnt/ssd2/robosnap
DISPLAY=:1 bash scripts/run_play_groot_wbc.sh \
    --scene-dir outputs/g1_ok \
    --mjcf outputs/g1_ok/mujoco/scene_g1.xml \
    --ply outputs/g1_ok/gs_bg_notripod.ply \
    --camera cap --fovy 55.21 --width 540 --height 720 \
    --ego-camera g1_head_pinhole --policy-port 5560 --policy-hz 20 \
    --vram-cap-gb 20
```

VRAM budget for this: 7.8 GB server + ~3 GB splat + margin ≈ 12 GB, comfortably
inside 31. Move the ELP publisher off 5560 first if it is up.

Useful flags: `--record run.mp4`, `--seconds N`, `--no-window`,
`--mode 3dgs_only|mujoco_only`, `--lang "…"` to change the task string.

### 4.4 What I changed in the harness, and why

`scripts/play_groot_wbc.py` was written for `outputs/video_scene` and needed
three changes for this scene. Backup at `play_groot_wbc.py.pre_g1ok`.

* `--camera NAME` — it used to *inject* a capture camera derived from
  `gravity_alignment.json`. That transform predates the hand-fit of `cap`
  against the photograph, so it looks from the wrong place. `--camera cap` uses
  the camera the MJCF already carries.
* `--fovy DEG` — with `--camera`, `real_intrinsics()` still returned the
  phone's fx/fy from `depth/camera.json`, which no longer describes the camera
  being rendered. The override recomputes `f = (H/2)/tan(fovy/2)` so the splat
  and the meshes are drawn at one focal length. Without this the two layers
  disagree and nothing lines up.
* `--no-policy` — `Groot.__init__` raises `SystemExit` if the server does not
  ping, so there was no way to run the WBC alone. Now the ego renderer is only
  built when a policy is attached, arms stay at the default stance, and the
  balance net runs.

### 4.5 Known limits of the policy route

* The checkpoint sorts **red/green/blue cubes**. This scene has an apple and a
  basket, so it is out of distribution: expect it to reach at nothing in
  particular. The pipeline is correct; the task is absent.
* GR00T's per-hand 7-vector is applied in the robot model's hand-group order
  (index, middle, thumb), not the MJCF's joint order (thumb first). Arms are
  unambiguous; the finger mapping is inferred and would show up as odd grasps.
* The ego view is a 60 deg pinhole. Training data came from the gear_sonic
  camera server, so framing is not identical — and the real robot's camera is
  the 180 deg ELP fisheye, which is a further mismatch. If the policy behaves
  strangely, this is the first thing to suspect.

---

## 5. The narrative — what was done, in order

### 5.1 The live MuGS viewer

`scripts/mugs_viewer.py` was written to drive MuGS's own `GaussianSensor`
rather than `hybrid_viewer.py`'s numpy compositing. It orbits by writing the
model camera directly and calling `mj_forward`.

**Pinning the floating base.** Holding the actuators at their rest angles is
*not* enough to keep the robot standing: measured, the pelvis still drops
0.792 → 0.189 m in two seconds, identical to leaving `ctrl` at zero. A
humanoid is not balanced by a PD hold. The viewer therefore pins the base
kinematically each step:

```python
data.qpos[base_adr:base_adr + 7] = base_q
data.qvel[base_dof:base_dof + 6] = 0.0
```

The WBC (§4) is the real answer — it balances the robot instead of freezing it.

### 5.2 Object/background overlap — Re3Sim's rule

The question was why the mesh objects and the splat's copies of the same
objects fought each other. Re3Sim does not remove objects from the background;
it *substitutes semantically*:

```python
mix_mask = table_mask | ground_mask
np.where(mix_mask, gaussians, sim_only_image)
```

No depth test — the splat simply wins on those surfaces. That was implemented
in `third_party/MuGS/src/mugs/sensors/gaussian_sensor.py` as a
`splat_geom_ids` config, plus `"seg_ids": seg_ids` in the hybrid dict:

```python
splat_painted = np.zeros_like(robot_mask, dtype=bool)
if self.cfg.splat_geom_ids:
    for gid in self.cfg.splat_geom_ids:
        splat_painted |= seg_ids == gid
    if self._last_bg_alpha is not None:
        splat_painted &= self._last_bg_alpha > 0.5
    robot_mask = robot_mask & ~splat_painted
```

Reached from the command line as `--splat-geoms round_wooden_table`.

### 5.3 The ego view was black — and the cause was not what it looked like

81.5% of the robot's fisheye was black. Tilt, compositing and the mesh were all
suspected and all cleared by measurement. The cause was **coverage**:
`bg_lm.ply` came from a single video clip via Lyra's feed-forward path, so
there simply were no gaussians where the robot was looking.

The fix was to train a proper wide 3DGS model on the full COLMAP reconstruction:

```
gs_wide.ply    3,592,082 gaussians
               PSNR 27.32 dB   SSIM 0.9209   sharpness 0.887
               13.2 min, 6.7 GB peak
```

Result (F13):

| | robot visible | ego fisheye black | doubled objects |
|---|---|---|---|
| `bg_lm.ply` | 25.9% | 81.5% | yes |
| `gs_bg_clean.ply` | 30.2% | **15.8%** | no |

Robot-eye coverage went 14.9% → 81.7%. This is the single most valuable fix in
the whole exercise.

### 5.4 Scale is unobservable from one view — the mistake worth remembering

Scaling a scene about the camera centre leaves every pixel exactly where it
was: the projection cancels `s`. NCC was measured identical for every scale
from 0.15 to 0.73. This is geometry, not a bug, and it is why fitting the
apple's *image position* gave a correct-looking picture with the room 37 cm out
of depth.

Depth does not cancel — `scripts/fit_splat_depth.py` fits `s` against MuJoCo's
depth buffer on surfaces both representations describe (table, apple, basket).

**Then I talked myself out of the right answer.** The apple's diameter implied
scale 0.383; I dismissed it with a hand-wave about Gaussian kernel inflation
and used the depth fit's 0.5075. The floor then landed at −0.257 — a 25.7 cm
float — and the user spotted it immediately: *"the robot is not on the floor
like it seems to be a bit up in the air."* The apple was right. Lesson: a
direct physical measurement beats an optimiser's residual.

### 5.5 Yaw: trust your eyes

A round table carries no yaw information, so yaw had to come from the image.
Five automated objectives gave five different answers (F14):

```
full-frame NCC      340.25
wall-only NCC        92.50
apple position      273.25
depth agreement     309.25
window brightness   353.75
```

Each locks onto a feature that repeats or mirrors elsewhere in the room. The
only reliable automatic discriminator was the **mirror test** — the apple sits
left of the basket in the photograph, which is only true from the correct side
of the room — and that brackets the answer to 273–310 deg. Inside that band the
user picked **268 deg** by eye, and that is what is baked in.
`scripts/tune_splat_yaw.py` exposes exactly that one knob.

### 5.6 The table dimensions closed the loop

The measured table — **74.5 cm high, 85.0 cm diameter** — was the key that
unlocked the scale. The reconstruction had the aspect ratio wrong by about 13%,
so no single factor could fix both height and diameter; the mesh scale had to
go non-uniform:

```
scale 1.18372 uniform  ->  1.088566 1.088566 1.285978     (18 mesh entries)
height    71.27 -> 74.47 cm   (real 74.50)
diameter  92.43 -> 84.39 cm   (real 85.00)
```

Objects were re-seated on the raised tabletop (58 contacts at rest, settling to
19 after 400 steps), and the splat re-pinned. The first re-pin, anchored on the
tabletop alone, put the floor at +0.0413 — pinning the tabletop shifts the
floor. Correcting the scale so *both* land:

```
0.3731 * 0.7447/0.7007 = 0.3965
```

which gives floor −0.0040 and tabletop +0.7425. Both within 4 mm, the first
time both constraints have been satisfied at once.

### 5.7 The camera swap

The G1's stock head camera is an Intel RealSense D435i. The real robot here
carries an **ELP-USB3DGS1200P01-L180**, so the D435i was replaced at the same
mount position:

```xml
<camera name="elp_left"  fovy="140" resolution="1600 1200" pos="0.08  0.03 0.35" .../>
<camera name="elp_right" fovy="140" resolution="1600 1200" pos="0.08 -0.03 0.35" .../>
<camera name="head_pinhole" fovy="60" resolution="640 480" pos="0.08 0 0.35" .../>
```

MuJoCo cameras are strictly rectilinear pinhole, so a 180 deg fisheye cannot be
rendered directly. `scripts/fisheye_render.py` renders a **cubemap** — five
faces at 90 deg — and remaps it to an equidistant fisheye, `r = f·θ`:

```python
FACES = {"front": _face((1,0,0),(0,1,0),(0,0,1)),
         "right": _face((0,0,1),(0,1,0),(-1,0,0)),
         "left":  _face((0,0,-1),(0,1,0),(1,0,0)),
         "up":    _face((1,0,0),(0,0,1),(0,-1,0)),
         "down":  _face((1,0,0),(0,0,-1),(0,1,0))}
```

The two lenses are published as **separate ZMQ streams** — PUB on 5560, topics
`elp/left` and `elp/right`, multipart `[topic, header json, JPEG]`, sharing a
`frame_id` and `stamp_ns` so a subscriber can prove they are synchronous. The
header carries `f_px_per_rad`, `fov_h_deg`, `baseline_m` and `projection`.

The 60 mm baseline is a **placeholder**, not a measurement.

### 5.8 The G1 model was vendored

The scene used to load the robot from `/mnt/ssd2/simulation_ws/modinoid_sim/robot/g1/`.
It is now copied byte-identical into `assets/robots/g1/` —
`g1_29dof_with_hand.xml`, `g1_29dof.xml`, `g1_29dof_3d.xml`, 50 meshes and a
README — and `scripts/add_robot.py` defaults to it. **Caveat:** `MjSpec.attach`
inlines the robot, so an already-built scene does not read these files again.
Any robot edit has to be applied to `scene_g1.xml` as well, or rebuilt.

---

## 6. Rebuilding the background chain

Already done; `gs_bg_notripod.ply` is the result. If it ever needs redoing:

```bash
# a) train the wide splat (13 min, 6.7 GB peak)
setsid nohup $LYRA scripts/train_gs.py \
    --colmap outputs/g1_ok/roommesh/dense \
    --image-dir outputs/g1_ok/roommesh/dense/images \
    --out outputs/g1_ok/gs_wide.ply --iters 30000 --long-side 1200 \
    --holdout 12 --vram-limit-gb 26 \
    > outputs/g1_ok/gs_wide_train.log 2>&1 < /dev/null & disown

# b) pick yaw by eye — do not trust the automated objectives (F14)
$SIM scripts/tune_splat_yaw.py --ply outputs/g1_ok/gs_wide.ply --scale 0.3965 --yaw 268

# c) bake the similarity into the PLY
$SIM scripts/apply_splat_transform.py --ply outputs/g1_ok/gs_wide.ply \
    --out outputs/g1_ok/gs_bg.ply \
    --scale 0.3965 --yaw 268.00 --min-opacity 0.02 \
    --world-table -0.055 0.724 0.7447

# d) clear the tabletop volume so objects do not appear twice
$SIM scripts/prune_splat.py --ply outputs/g1_ok/gs_bg.ply \
    --mjcf outputs/g1_ok/mujoco/scene_g1.xml \
    --out outputs/g1_ok/gs_bg_clean.ply --keep-below 0.03

# e) cut the tripod out of the sub-floor shell (see 7.1 - this is the step that
#    --keep-below in (d) was quietly undoing)
$SIM - <<'EOF'
import numpy as np
from plyfile import PlyData, PlyElement
v = PlyData.read("outputs/g1_ok/gs_bg_clean.ply")["vertex"]
X = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
lum = (np.clip(0.5 + 0.2820948 * np.stack([v[f"f_dc_{i}"] for i in range(3)], 1),
               0, 1) @ [0.299, 0.587, 0.114])
r = np.linalg.norm(X[:, :2] - [-0.058, 0.7255], axis=1)
keep = ~((lum < 0.25) & (X[:, 2] > -0.14) & (X[:, 2] < 0.012) & (r < 0.95))
arr = np.empty(int(keep.sum()), dtype=v.data.dtype)
for n in v.data.dtype.names:
    arr[n] = v[n][keep]
PlyData([PlyElement.describe(arr, "vertex")], text=False).write(
    "outputs/g1_ok/gs_bg_notripod.ply")
print(f"kept {keep.sum():,} of {len(X):,}")
EOF
```

The COLMAP model must be at `<colmap>/colmap/sparse/0`. It is symlinked; if you
rebuild the workspace, redo it:

```bash
cd outputs/g1_ok/roommesh/dense && mkdir -p colmap/sparse && ln -sfn ../../sparse colmap/sparse/0
```

Evaluate at one fixed resolution — comparing scores from different resolutions
measures the resolution, not the model:

```bash
$LYRA scripts/eval_splats.py --colmap outputs/g1_ok/roommesh/dense \
    --ply outputs/g1_ok/gs_wide.ply --label gs_wide --eval-long-side 1200 --holdout 12
```

The prune that actually worked is a numpy filter, not the `--body` flags: clear
the whole cylinder over the tabletop (r = 0.90 m, z = 0.62…1.20), because the
mesh table and mesh objects own that volume entirely. The `--body` route chases
each object's convex hull and **misses the basket**, because the splat's copy of
it sits where the mesh's does not.

---

## 7. Open items

### 7.1 The tripod — found and cut

**Corrected.** I previously told you this was unfixable by pruning. That was
wrong, and the reason I was wrong is worth writing down.

The background table's floor frame reads in the render as a black tripod on the
carpet. I had measured its gaussians as sharing the carpet's height band and
concluded there was no separation to exploit. What I had never done was render
`--mode 3dgs_only` and `--mode mujoco_only` from `cap` and compare them. Doing
that took two minutes and showed immediately that the tripod is in the *splat*,
that the table's legs and centre post are in the *mesh*, and — the actual point —
that the tripod's gaussians sit at

```
z  -0.139 .. +0.004   (median -0.033)     r 0.00 .. 0.95 from the table centre
```

i.e. in a thin shell **below** the floor plane (splat floor is z = -0.0040). Every
one of my eight failed prunes cut the band immediately *above* the floor —
`z 0.008 .. 0.14` — which is clean carpet, and left the tripod untouched. Worse,
`prune_splat.py --keep-below 0.03` was explicitly *protecting* the tripod the
whole time. I was cutting good carpet and preserving the thing I was trying to
delete, which is exactly why every attempt produced a hole and no improvement.

The cut that works, colour-gated so the carpet at the same height survives:

```python
kill = (lum < 0.25) & (X[:,2] > -0.14) & (X[:,2] < 0.012) & (r < 0.95)
# 81,269 of 2,222,113  (3.66%)  ->  outputs/g1_ok/gs_bg_notripod.ply
```

The solid black tripod is gone. Verified in the hybrid from `cap` and in the ego
fisheye — 93.6% covered, **0.0% unfilled inside the lens circle**, so the cut
opened no hole in the robot's view.

**What is left, honestly:** a faint grey outline where the tripod's arms were.
That residue cannot be cut by luminance, and this time the measurement says so
rather than my intuition: in the tripod's own image footprint the remaining
gaussians run median lum 0.46–0.58, against reference carpet at median 0.39.
The residue is *brighter* than the carpet around it — it is carpet-coloured, so
any threshold that removes it removes the carpet first.

**The carpet graft, and why it fails.** I also had this wrong. I had said
grafting smears because gaussians carry viewpoint-specific orientations — but
`apply_splat_transform.py` already drops this PLY to DC-only, so there is no
view-dependent colour left to smear. I tried it properly: cut the shell
colour-blind, then remap a clean donor ring (r 1.50–2.60) inward over the hole.
It smears badly, for a completely different reason — **the remap moves each
gaussian's mean but not its covariance**, so compressing a 1.1 m ring into a
0.9 m disc stretches every kernel into a streak, and the angular convergence at
small radius makes it worse. A correct graft would have to rescale each
gaussian's covariance by the local Jacobian of the remap, and would still be
25× too sparse: the donor ring holds 29,820 gaussians against a 753,507-gaussian
hole. Not worth it. The attempt was deleted.

**If you want the residue gone too**, the route is still mask → inpaint →
retrain on the 450 source frames: then the tripod was never in the splat, so
there is no ghost and no hole. About 30 min masking and inpainting plus 13 min
training. Everything needed for the retrain is in §6.

### 7.2 Smaller open items

* **The SH-drop cost is unmeasured.** 27.32 dB / SSIM 0.9209 belongs to
  `gs_wide.ply` *with* its 45 `f_rest` bands (degree 3).
  `apply_splat_transform.py` drops them to DC, because rotating degree-3 SH
  needs Wigner D-matrices and getting that subtly wrong costs more than the
  bands are worth. The loss has never been quantified.
* **The ELP baseline is a placeholder** — 60 mm, not measured on the real
  camera.
* **The DDS WBC route is blocked** — `unitree_sdk2py` and `gear_sonic` are not
  installed in any env here (§4.1).
* **Prune radius** was not re-tuned after the dimension change.
* **The room mesh was abandoned.** The MVS mesh has a 2.0 m² floor, no tabletop
  and non-planar walls; registration was not worth continuing. ICP failed every
  way it was tried (F12): scaled ICP collapsed to 0.29×, free-rotation ICP
  tilted 80 deg, yaw-locked plateaued at 34 cm.

---

## 8. Things I got wrong, so you do not repeat them

1. **Grepping a process list inline.** The grep matches its own command line, so
   everything looks alive. Snapshot to a file first. This caused repeated false
   "still running" reports while a job had been dead for 40 minutes.
2. **`nohup … &` without `setsid`.** Jobs died when a tool call hit its timeout.
3. **Reading `mesh_vert` as if it were in the OBJ's frame.** It is stored in a
   principal-axis frame. I concluded the room mesh was 2.8× oversized; the OBJ
   actually spans 3.55 m.
4. **Quoting metrics computed on different masks as comparable.**
   `fit_splat_depth.py`'s 32.6%-within-3 cm used the shared-surface mask
   (23.8%); `mugs_render`'s 19.6% used the full foreground (30.2%). Different
   denominators, not comparable.
5. **Trusting an optimiser over a tape measure** (§5.4).
6. **Diagnosing before rendering.** I claimed the depth test was eating the mesh
   legs; it was the orbit viewpoint's framing. From `cap` it is 30.2% either
   way. One mesh-only render would have settled it.
7. **Auto-framing the orbit target from body positions.** Fitting the scene then
   moved the camera, silently changing the viewing angle between runs. Fixed
   with explicit `--look-at/--distance/--azimuth/--elevation`.
