# Unitree G1 29-DoF (vendored)

Copied here so the robot can be edited in this repo instead of in whichever
`modinoid_sim` checkout happens to be on the machine. There are five copies of
this robot on this box and they are **not** interchangeable - joint ordering
differs, and the FixStand gains in `scripts/apple_scene_demo.py` are tied to
the 29-DoF ordering of these files.

    source   /mnt/ssd2/simulation_ws/modinoid_sim/robot/g1/
    copied   2026-09-08, byte-identical to the source at that date

| file | notes |
|---|---|
| `g1_29dof_with_hand.xml` | what the scenes use - three-finger hands, 46 bodies / 44 joints / 43 actuators |
| `g1_29dof.xml` | same skeleton, no hands |
| `g1_29dof_3d.xml` | variant kept for reference |
| `meshes/` | the 50 STLs these three MJCFs reference (of 65 in the source tree) |

`scripts/add_robot.py` defaults to `g1_29dof_with_hand.xml` here, so
editing it changes every scene built from now on.

## Careful: built scenes do not read these files

`add_robot.py` uses `MjSpec.attach`, which **inlines** the robot. A built scene
such as `outputs/g1_ok/mujoco/scene_g1.xml` contains the bodies directly and
carries its own copy of the meshes in `outputs/g1_ok/mujoco/assets/`. Editing
the MJCF here has no effect on a scene that already exists - re-run
`add_robot.py` to pick the change up.

The `--prefix g1_` that `add_robot.py` applies is what lets `mugs_render.py`
and `scripts/mugs_viewer.py` tell robot geoms from scene geoms, so keep it.

## Provenance of the standing pose

The FixStand pose and gains come from
`unitree_rl_lab/deploy/robots/g1_29dof/config/config.yaml`, which matches the
joint ordering of these files. Check the ordering before reusing them with any
other G1 copy on this machine.
