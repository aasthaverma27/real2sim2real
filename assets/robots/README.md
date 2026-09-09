# Robot models

**This directory is intentionally almost empty.** Robot models are not
committed — they belong to their manufacturers, and you will be bringing your
own robot anyway.

Drop a robot here, then describe it in `configs/robots/<name>.yaml`. Nothing in
the codebase is specific to any particular robot; see the README's
"Add your own robot".

```
assets/robots/
  <your_robot>/
    <your_robot>.xml        MuJoCo MJCF
    meshes/                 whatever <compiler meshdir="..."> points at
```

`meshes/` directories are gitignored, so a model you drop here stays local.

## Requirements a model has to meet

Any MJCF works for **rendering**. To drive one with a deployed whole-body
controller, the model must also satisfy what that controller assumes — for the
GR00T WBC that is:

* a free joint on the root body, named in the descriptor's `sim.base_joint`;
* the root body and torso named as the controller expects (`pelvis`,
  `torso_link` for the G1) and **not** namespaced;
* the free joint compiling to joint 0, with the robot's hinges at 1..N and
  nothing in front of them — controllers commonly index state as
  `qpos[joint_id + 6]`.

`scripts/realsim/build_wbc_scene.py` enforces the last two at build time rather
than letting a bad scene drive the wrong joints at 50 Hz.

## Getting a model

```bash
scripts/fetch_robot.sh --list
scripts/fetch_robot.sh unitree_g1
scripts/fetch_robot.sh --from /path/to/your/robot.xml --name my_robot
```

Good sources of ready-made MJCFs:

| source | licence | notes |
|---|---|---|
| [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | Apache-2.0 (per-model notices) | curated, well-tuned; many humanoids and arms |
| manufacturer SDK | varies | usually the most accurate joint limits and inertias |
| your own URDF | yours | convert with `mujoco.MjSpec` or `urdf2mjcf` |

Check each model's own licence before redistributing it. Menagerie models carry
per-model `LICENSE` files, and they are not all the same.
