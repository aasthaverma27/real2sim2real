# Playing a GR00T checkpoint in the photoreal scene

`scripts/play_groot_wbc.py` runs a trained GR00T checkpoint and the GEAR WBC in
one process with MuJoCo and MuGS's `GaussianSensor`, so the robot acts in the
reconstructed room and the composite is displayed live.

    DISPLAY=:1 bash scripts/run_play_groot_wbc.sh --scene-dir outputs/video_scene

## Why it has to be one process

MuGS renders whatever `mj_data` its own process holds - `sensor.render(model,
data, camera)` has no step function and no notion of another simulator. Their
showcase works the same way: AndroidTwin's `at-eval` runs policy plus whole-body
controller in-process and passes each frame to `GaussianSensor`. There is no
viewer and no checkpoint playback anywhere in the MuGS repo.

The WBC is not optional either. The checkpoint's action space is

    left_arm 7, right_arm 7, left_hand 7, right_hand 7, waist 3,
    base_height_command 1, navigate_command 3

with **no leg actions**: it steers the base with a velocity command, and the
Balance/Walk ONNX nets turn that into leg targets. So the loop is

    g1_ego_view -> GR00T :5560 -> arms + hands -> joint targets
                              \-> navigate_command + base_height
                                     -> WBC Balance/Walk -> legs + waist
                                     -> PD torques -> mj_step
                                     -> GaussianSensor.render() -> window

Neither policy is modified: the checkpoint is served by the stock
`run_gr00t_server`, the WBC nets are the shipped ONNX files.

## Running it

### 1. The policy server (the only other process needed)

```bash
cd /mnt/ssd2/Isaac-GR00T
export NO_ALBUMENTATIONS_UPDATE=1 HF_HOME=/mnt/ssd2/huggingface_cache \
       PYTHONPATH=/mnt/ssd2/Isaac-GR00T
setsid nohup /mnt/ssd2/dsrl/.venv/bin/python -u -m gr00t.eval.run_gr00t_server \
  --model_path /mnt/ssd2/Isaac-GR00T/checkpoints/red_green_blue_cube_sorting_in_sim/checkpoint-22000 \
  --embodiment_tag unitree_g1_full_body_with_waist_height_nav_cmd \
  --device cuda --host 0.0.0.0 --port 5560 > /tmp/gr00t_server.log 2>&1 < /dev/null &
```

Wait for `✓ Server ready` (~30 s, 7.8 GB VRAM). `setsid` matters: without it the
server dies with the shell that launched it.

The sim, the WBC container and the inference client are **not** used - this
harness is those three, in one process.

### 2. Live window

```bash
cd /mnt/ssd2/robosnap
DISPLAY=:1 bash scripts/run_play_groot_wbc.sh --scene-dir outputs/video_scene
```

ESC quits. Useful flags:

| flag | effect |
|---|---|
| `--record run.mp4` | write what is displayed |
| `--seconds 60` | stop after N seconds |
| `--no-window` | record only, for headless runs |
| `--mode 3dgs_only` / `mujoco_only` | inspect one layer |
| `--width/--height` | display size (default 440x780) |
| `--policy-hz` | GR00T query rate, default 20 (the dataset rate) |

### 3. Headless capture (what produced the first video)

```bash
bash scripts/run_play_groot_wbc.sh --scene-dir outputs/video_scene \
    --no-window --seconds 30 --record outputs/video_scene/groot_wbc.mp4
```

## Environment

`robosnap-sim` has gsplat/torch/mujoco but not the client libraries. One-time:

```bash
/mnt/ssd2/conda/envs/robosnap-sim/bin/python -m pip install \
    pyzmq msgpack msgpack-numpy onnxruntime "transformers==4.57.3"
```

`transformers` is only needed because `gr00t.policy.__init__` imports
`Gr00tPolicy`, which pulls it in even though the client itself never uses it.

## Known limits

* The checkpoint sorts red/green/blue cubes; a reconstruction without cubes is
  out of distribution, so it reaches at an empty table. The pipeline is correct;
  the task is not present.
* GR00T's per-hand 7-vector is applied in the robot model's hand-group order
  (index, middle, thumb), not the MJCF's joint order (thumb first). Arms are
  unambiguous; finger mapping is inferred and would show up as odd grasps.
* The ego camera is rendered at 640x480 from `g1_ego_view`; the training data
  came from the gear_sonic camera server, so framing is not identical.
