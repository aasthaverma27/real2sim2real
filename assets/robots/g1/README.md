# Unitree G1 (29 body DoF + 2 × 7 hand DoF)

| file | what it is |
|---|---|
| `g1_cad.xml` | **The robot every scene is built with** (`--robot g1_cad`, the default). Identical to `g1_29dof_with_hand.xml` except for the five camera poses: their positions come from the CAD of our camera mounts, and the mount parts themselves are drawn. |
| `g1_29dof_with_hand.xml` | Same robot with the earlier hand-measured camera positions. Kept for comparison. |
| `g1_29dof.xml` | Same skeleton, no hands. |
| `cad/manifest.json` | Which CAD part is attached to which robot link. |
| `meshes/cad/*.stl` | The camera-mount parts (head camera mount and height adjuster, the two wrist camera rings and mounts), exported from the Fusion assembly. Committed. |
| `meshes/*.STL` | Stock Unitree meshes. **Not committed**: run `bash scripts/fetch_g1_meshes.sh`. |

Both hand-equipped models use the hand actuators in **thumb-first** order
(`thumb_0..2`, then the two fingers), 43 actuators in total. This is the order
the deployed whole-body controller drives correctly;
`scripts/realsim/verify_scene.py` checks it.

## Cameras

| MJCF camera | parent link | `fovy` (deg) | resolution | source |
|---|---|---|---|---|
| `elp_left`, `elp_right`, `head_pinhole` | `torso_link` | 53.35 | 768 × 432 | checkerboard calibration (`calib/stereo_*.yaml`) |
| `ego_view_left_mono` | `left_wrist_yaw_link` | 63.34 | 1280 × 720 | checkerboard calibration (`calib/hand_left.yaml`) |
| `ego_view_right_mono` | `right_wrist_yaw_link` | 63.34 | 1280 × 720 | checkerboard calibration (`calib/hand_right.yaml`) |

Intrinsics are measured (see [docs/camera_calibration.md](../../../docs/camera_calibration.md)).
Positions come from CAD. The head pitch (56.7° below horizontal) was measured
with solvePnP against a flat checkerboard. No hand-eye calibration was done,
so treat the wrist-camera orientations as approximate.

## Built scenes do not read these files

`add_robot.py` uses `MjSpec.attach`, which **inlines** the robot. A built scene
(`outputs/<scene>/mujoco/scene_g1.xml` and `wbc/scene_wbc.xml`) contains the
bodies directly and carries its own copy of the meshes. Editing an MJCF here
has no effect on an existing scene until you rebuild those two stages:

```bash
bash scripts/build_sim_scene.sh --scene-dir outputs/myroom --only robot --force \
    --robot-at X Y --robot-yaw RAD
bash scripts/build_sim_scene.sh --scene-dir outputs/myroom --only wbc --force \
    --robot-at X Y --robot-yaw RAD
```

To move only the camera rig between already-built scenes, without a rebuild,
use `scripts/realsim/port_cad_rig.py`.

The `g1_` prefix that `add_robot.py` applies is how the renderers tell robot
geoms from scene geoms, so keep it.
