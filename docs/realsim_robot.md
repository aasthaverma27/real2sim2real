# Putting a robot in a RoboSnap scene

Reconstruct a scene from pictures, stand a robot in it, drive it with a
whole-body controller, and render it against the reconstruction's Gaussian
splat. Nothing here is G1-specific: every robot-dependent value lives in
`configs/robots/<name>.yaml`, and the scripts read it.

```
pictures ──► reconstruct ──► export MJCF ──► place robot ──► merge for WBC
                                  │              │               │
                                  └──────────────┴───────────────┴──► run
                                                                       ├─ sim + controller
                                                                       ├─ hybrid viewer
                                                                       └─ robot camera
```

## 0. Environment

```bash
cd /mnt/ssd2/robosnap
export PATH=/mnt/ssd2/conda/envs/robosnap-sim/bin:$PATH
export PY=/mnt/ssd2/conda/envs/robosnap-sim/bin/python
export MUJOCO_GL=egl                # glfw when you want an interactive viewer
```

`PATH` is not optional for anything touching gsplat: it JIT-compiles CUDA on
first use and needs `ninja` from that env. The first render therefore takes a
minute or two; later ones are fast.

## 1. Pictures in, scene out

```bash
INPUT_IMAGE=inputs/YOUR_IMAGE.png \
OUTPUT_DIR=outputs/myscene \
OBJECT_PROMPTS="the round wooden table,the red apple,the woven brown basket" \
SF_REAL2SIM_COLLISION_METHOD=coacd \
  bash scripts/run_object_pipeline.sh
```

One image is enough for the objects. The background splat wants a video of the
same room — see `RUNBOOK_g1.md` §3 for that half, and note the two stages must
not run at once: SAM-3D peaks near 17 GB and the background stage near 18 GB.

Then physics properties and the MJCF:

```bash
$PY scripts/estimate_properties.py --scene-dir outputs/myscene --mode lookup
$PY scripts/export_mujoco.py --scene-dir outputs/myscene --static table
```

Reconstruction is scale-free, so set the absolute size from one measurement:

```bash
$PY scripts/rescale_scene.py \
    --scene outputs/myscene/mujoco/scene.xml \
    --output outputs/myscene/mujoco/scene_scaled.xml \
    --table-body obj0_round_wooden_table --table-height 0.745
```

## 2. Stand the robot in it

```bash
$PY scripts/add_robot.py \
    --scene-dir outputs/myscene \
    --scene-xml outputs/myscene/mujoco/scene_scaled.xml \
    --robot assets/robots/g1/g1_29dof_with_hand.xml \
    --at 0.0 1.37 --yaw -1.62 --strip-rangefinders \
    --output outputs/myscene/mujoco/scene_robot.xml
```

This namespaces the robot (`g1_*`) and is the scene to **render**.

## 3. Merge for the controller — the step that matters

A deployed WBC sim is strict about the robot, and a scene with the robot
attached *into* it breaks two invariants:

* bodies must keep their own names (`pelvis`, `torso_link`) and the base joint
  must be `floating_base_joint` — attaching into the scene namespaces them;
* joint state is read as `qpos[joint_id + 6]`, so the base joint has to be
  joint 0 and the hinges 1..N with nothing in front.

So the merge runs the other way: the **robot MJCF is the parent** and the scene
attaches into it, transformed so the robot's stance lands on the origin. This
is the mirror of step 2, and it is how the reference stack builds its scenes.

```bash
$PY scripts/realsim/build_wbc_scene.py \
    --scene-dir outputs/myscene --robot g1_29dof_with_hand \
    --scene-xml outputs/myscene/mujoco/scene_robot.xml --strip-prefix g1_ \
    --at 0.0 1.37 --yaw -1.62
```

`--strip-prefix g1_` removes the already-placed robot first, so the merge uses
the scene you actually tuned rather than the raw export. Both invariants are
checked and the script fails loudly rather than silently driving the wrong
joints at 50 Hz:

```
[wbc-scene] robot occupies joints 0..43, scene starts after
[wbc-scene] merged: nbody=49 nq=64 nu=43
Point the reference sim at it:
  ROBOT_SCENE: ".../mujoco/wbc/scene_wbc.xml"
```

Put that path in the reference sim's `ROBOT_SCENE`. Nothing about the
controller, its ONNX policies or its DDS loop changes.

## 4. Run it

### Controller only — the robot stands

```bash
MUJOCO_GL=glfw $PY scripts/realsim/run_wbc.py \
    --robot g1_29dof_with_hand \
    --mjcf outputs/myscene/mujoco/scene_robot.xml --prefix g1_ \
    --ply outputs/myscene/gs_bg.ply \
    --camera cap --width 540 --height 720 --viewer
```

`--viewer` opens MuJoCo's interactive viewer on the *same* `mj_data` as the
composite, so both windows show one simulation. Drop `--ply` and pass
`--mode mujoco_only` for a fast physics-only check.

The controller is not optional: a humanoid is not held up by a PD hold on its
rest pose. Measured on the G1 the pelvis still drops 0.792 → 0.189 m in two
seconds with the actuators holding, identical to leaving `ctrl` at zero.

### Controller + a policy on the arms

```bash
MUJOCO_GL=glfw $PY scripts/realsim/run_wbc.py \
    --robot g1_29dof_with_hand \
    --mjcf outputs/myscene/mujoco/scene_robot.xml --prefix g1_ \
    --ply outputs/myscene/gs_bg.ply --camera cap \
    --policy-host localhost --policy-port 5560 \
    --ego-camera g1_head_pinhole \
    --lang "pick up the red apple and place it in the basket"
```

Needs a checkpoint served by the stock `run_gr00t_server` on 5560.

### Hybrid viewer, on its own

```bash
$PY scripts/mugs_viewer.py \
    --scene-dir outputs/myscene \
    --mjcf outputs/myscene/mujoco/scene_robot.xml \
    --ply outputs/myscene/gs_bg.ply \
    --camera scene_view --capture-camera cap \
    --width 700 --height 933 --splat-geoms round_wooden_table
```

Add `--mirror 5601` to follow a DDS sim instead of stepping its own physics
(needs `scripts/wbc_state_relay.py`; see `docs/g1_full_stack.md`).

### Robot camera

Two streams over ZMQ, one per lens. MuJoCo cameras are strictly rectilinear, so
a fisheye is rendered as a 5-face cubemap and remapped to `r = f·θ`.

```bash
$PY scripts/elp_camera_publisher.py \
    --scene-dir outputs/myscene \
    --mjcf outputs/myscene/mujoco/scene_robot.xml \
    --ply outputs/myscene/gs_bg.ply \
    --port 5562 --width 640 --height 480 --fov-h 170 --fps 6

$PY scripts/elp_camera_subscriber.py --port 5562 --side both --topmost
```

Expect `pair synchronous, stamp delta 0.000 ms`. One still frame instead:
`scripts/fisheye_render.py --eye both -o /tmp/elp.png`.

## Adding another robot

Copy a descriptor and change the values — no code:

```bash
cp configs/robots/g1_29dof_with_hand.yaml configs/robots/myrobot.yaml
```

| field | what it is |
|---|---|
| `mjcf` | robot MJCF, repo-relative or absolute |
| `sim.base_joint` / `sim.root_body` | names the reference sim requires |
| `sim.n_robot_joints` | hinge count; the 0..N invariant is checked against it |
| `body_joints` | **canonical order the controller's observation is indexed by** — do not reorder |
| `hand_finger_order` / `hand_sides` | expands to the hand joint names |
| `wbc.balance_onnx` / `walk_onnx` | the two nets; balance runs when the velocity command is ~0 |
| `wbc.n_act` | joints the nets emit (legs + waist) |
| `wbc.kp` / `kd` / `default_pose` | per-joint, length = `len(body_joints)` |
| `cameras.*` | head mount, tilt, fisheye/pinhole intrinsics |

`run_wbc.py` checks the observation width against the net on startup:

```
the balance net wants 516 observations but the descriptor builds 480
(80 x 6). Check body_joints, wbc.n_act and wbc.obs_history.
```

so a miscounted descriptor fails immediately instead of producing garbage.

## Verified

Both scene shapes drive the same controller from one descriptor:

| scene | prefix | result |
|---|---|---|
| `wbc/scene_wbc.xml` (robot-parent merge) | none | 5,587 control steps, base 0.787 → 0.745 |
| `scene_g1.xml` (robot attached into scene) | `g1_` | 9,414 control steps, base 0.787 → 0.745 |

`0.745` against a commanded `0.74` is the balance net holding station, not a
collapse. Merge fidelity: table height identical to 0.1 mm across both scenes,
horizontal extent swapped as expected by the −92.9° re-origin.
