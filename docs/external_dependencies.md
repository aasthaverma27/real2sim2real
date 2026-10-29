# External dependency: NVIDIA GR00T-WholeBodyControl

Teleoperation and dataset recording run on NVIDIA's public
[GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl),
included as a **git submodule** at `external/GR00T-WholeBodyControl`, pinned to
commit `b042411`. It is the only external stack. Everything in sections 2–5 of
the README (scene building, calibration, the photoreal sim with the in-process
controller) works without it.

```bash
bash scripts/setup_external.sh --check     # what is present
bash scripts/setup_external.sh             # fetch, patch, model files, .venv_sim
cd external/GR00T-WholeBodyControl && ./decoupled_wbc/docker/run_docker.sh --install --root   # container image
```

## What we use from it

| part | path in the submodule | where it runs |
|---|---|---|
| whole-body controller (`run_g1_control_loop.py`, GEAR-WBC balance/walk policies) | `decoupled_wbc/control/main/teleop/`, `decoupled_wbc/sim2mujoco/resources/robots/g1/policy/` | container |
| Pico VR teleop (`run_teleop_policy_loop.py`, `pico_streamer.py`) | `decoupled_wbc/control/` | container |
| LeRobot exporter (`run_g1_data_exporter.py`) | `decoupled_wbc/control/main/teleop/` | container |
| dataset post-processing (`process_dataset.py`) | `gear_sonic/scripts/` | container, via `finalize_dataset.sh` |
| MuJoCo ↔ DDS bridge (`unitree_sdk2py_bridge.py`) | `gear_sonic/utils/mujoco_sim/` | host, in `.venv_sim` |
| G1 model and meshes | `gear_sonic/data/robot_model/model_data/g1/` | host (`fetch_g1_meshes.sh`) |
| Unitree SDK (`unitree_sdk2py`) | `external_dependencies/unitree_sdk2_python/` | host and container |

The container is NVIDIA's prebuilt image `nvgear/gr00t_wbc:latest`. Its
run script mounts the submodule at `/root/Projects/GR00T-WholeBodyControl`, and
the commands in [teleop_dataset.md](teleop_dataset.md) put that directory first
on `PYTHONPATH`, so the mounted, patched code is what runs.

`setup_external.sh` fetches only the Git LFS files the pipeline needs (the G1
meshes and URDF, the two ONNX policies, the Unitree SDK's binaries), about
55 MB, rather than the repository's full model set. It builds `.venv_sim`
with NVIDIA's own `install_scripts/install_mujoco_sim.sh`.

## Our changes: one patch

`patches/GR00T-WholeBodyControl_sim_teleop.patch`, applied by
`setup_external.sh`:

| change | file | why |
|---|---|---|
| ZMQ camera client (new) | `decoupled_wbc/control/sensor/vla_image_client.py` | the exporter records the three cameras our camera server publishes; re-subscribes when the server restarts |
| exporter uses it | `run_g1_data_exporter.py` | plus a diagnostic that says *why* no image has arrived |
| grasp smoothing (new) | `decoupled_wbc/control/utils/hand_grasp_smoother.py`, `run_g1_control_loop.py`, `configs.py` | rate-limits the right-hand grasp so the executed and the recorded action match |
| DDS domain 99 | `g1_29dof_gear_wbc.yaml` | the sim's domain; the real robot is on 0 |
| three-camera defaults | `configs.py` | head + two wrists; the stereo pair is not published |
| camera probe (new) | `decoupled_wbc/scripts/zmq_probe.py` | checks the camera server before recording |
| resting elbow override | `g1_supplemental_info.py` | `G1_DEFAULT_ELBOW` (rad), read only when set, bends both elbows in the default pose so the resting hands clear tall objects (`scenes/new_table_env`); `start.sh` sets it from the scene's `spawn_pose.json` |
| π0.5 policy loop (new) | `decoupled_wbc/control/main/pi05/` | runs an openpi policy server's actions through the WBC on the real robot or in the sim; see [finetune_pi05.md](finetune_pi05.md) |

To move to a newer upstream commit, change the pin in
`scripts/setup_external.sh` and `scripts/register_submodules.sh`, check that
the patch still applies (`git -C external/GR00T-WholeBodyControl apply --check
../../patches/GR00T-WholeBodyControl_sim_teleop.patch`), and test.

## Licences

GR00T-WholeBodyControl's source code is under Apache-2.0, and its model
weights (the ONNX policies) are under the NVIDIA Open Model License; see its
`LICENSE` and `legal/`. The Unitree SDK is BSD-3-Clause. Our patch is part of
this repository and under its licence. Model weights are never committed here.
