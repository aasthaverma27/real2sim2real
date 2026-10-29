# Fine-tuning π0.5 on simulated G1 data

This guide trains a [π0.5](https://github.com/Physical-Intelligence/openpi) policy
on episodes recorded in a real2sim2real scene, then runs it on the robot or back in
the simulator. It is the procedure behind the released
[`pi05-g1-fruit-v2`](https://huggingface.co/aasthaverma27/pi05-g1-fruit-v2) checkpoint.

The training code is a fork of openpi, checked out as the `external/openpi`
submodule. It adds:

- the G1 + Dex3 policy transforms (`src/openpi/policies/g1_dex3_policy.py`)
- the `pi05_g1_*` train configs (`src/openpi/training/config.py`)
- validation loss during training
- a norm-stats script for these datasets
- checkpoint tests and offline evaluation scripts

Everything else is upstream openpi.

## 1. Set up openpi

```bash
git submodule update --init external/openpi
cd external/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Full fine-tuning at batch size 32 needs about 70 GB of GPU memory. The released
run used one RTX PRO 6000 (96 GB). With less memory, lower `--batch-size`.

## 2. Record and prepare a dataset

Record demonstrations as in [teleop_dataset.md](teleop_dataset.md), then merge
the sessions into one LeRobot v2.1 dataset:

```bash
scripts/realsim/finalize_dataset.sh fruit-pnp <session> [<session> ...]
```

Hold out a few episodes per task for validation:

```bash
export HF_LEROBOT_HOME=$HOME/lerobot_datasets
python scripts/pi05/split_dataset.py $RS_DEPLOY_ROOT/outputs/fruit-pnp \
    --out $HF_LEROBOT_HOME/fruit-pnp --val-per-task 5
# -> $HF_LEROBOT_HOME/fruit-pnp/train, $HF_LEROBOT_HOME/fruit-pnp/val
```

The dataset keeps the exporter's 43-D whole-body `observation.state` and `action`.
The policy uses the 28 upper-body dimensions (indices 15 to 42), ordered
`[left arm 7, left hand 7, right arm 7, right hand 7]`. The legs and waist are
left to the whole-body controller. The task string of each episode becomes the
language prompt.

## 3. Add a train config

In `external/openpi/src/openpi/training/config.py`, copy the `pi05_g1_fruit_v2`
entry, give it a new name and point it at your splits:

```python
TrainConfig(
    name="pi05_g1_my_task",
    model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
    data=LeRobotG1Dex3DataConfig(
        repo_id="fruit-pnp/train",
        base_config=DataConfig(prompt_from_task=True),
    ),
    val_repo_id="fruit-pnp/val",
    ...  # optimizer, schedule and step counts as in pi05_g1_fruit_v2
),
```

## 4. Train

```bash
cd external/openpi
export HF_LEROBOT_HOME=$HOME/lerobot_datasets

uv run scripts/compute_norm_stats_g1_dex3.py --config-name pi05_g1_my_task
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_g1_my_task --exp-name run1
uvx tensorboard --logdir checkpoints/pi05_g1_my_task      # train and validation loss
```

Add `--resume` to continue an interrupted run. The configs also log to Weights &
Biases; `--no-wandb-enabled` turns that off.

## 5. Check a checkpoint

```bash
# loads, output shape, finite actions, action space, joint ranges; exit code 1 on failure
uv run scripts/test_g1_dex3_checkpoint.py --config-name pi05_g1_my_task \
    --checkpoint-dir checkpoints/pi05_g1_my_task/run1/10000

# action error on the held-out episodes, per joint group and horizon
uv run scripts/eval_g1_dex3_offline.py --config-name pi05_g1_my_task --dataset fruit-pnp \
    --checkpoint-dir checkpoints/pi05_g1_my_task/run1/10000 --split val
```

## 6. Run the policy

Start the policy server on the GPU machine:

```bash
uv run --project external/openpi python scripts/pi05/serve_pi05.py \
    --config pi05_g1_my_task --checkpoint-dir external/openpi/checkpoints/pi05_g1_my_task/run1/10000
```

To use the released checkpoint instead, download one step and serve it with the
default config:

```bash
huggingface-cli download aasthaverma27/pi05-g1-fruit-v2 --include "10000/*" \
    --local-dir checkpoints/pi05_g1_fruit_v2
uv run --project external/openpi python scripts/pi05/serve_pi05.py \
    --checkpoint-dir checkpoints/pi05_g1_fruit_v2/10000
```

The policy loop runs inside the whole-body-controller container, next to the
WBC. It needs the openpi client package once:

```bash
pip install "openpi-client @ git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=packages/openpi-client"
```

Then, with the WBC running:

```bash
# one action chunk printed next to the measured state, nothing sent to the robot
python decoupled_wbc/control/main/pi05/run_pi05_policy.py --probe --policy-host <gpu-host>

python decoupled_wbc/control/main/pi05/run_pi05_policy.py --policy-host <gpu-host> \
    --prompt "pick up the mango and place it in the basket"
```

The same loop drives the real robot and the simulator. For the simulator, start
the scene with its cameras served and the fruit on the plate:

```bash
scripts/realsim/start.sh --scene scenes/new_table_env --vla --mango --dragon-fruit
```

On the real robot, pass `--camera-host <robot-ip>`. Each rollout is saved under
`inference_recordings/`: the head-camera video, every action chunk and a per-tick
CSV.

## Pitfalls

**Arm actions are deltas, hand actions are absolute.**
- The configs train with `DeltaActions(make_bool_mask(7, -7, 7, -7))`.
- `serve_pi05.py` sends the arm deltas as the model outputs them, and the client
  adds the current state once (`--arm-action-mode delta`, its default).
- If the server is started with `--arm-actions absolute`, the client must use
  `--arm-action-mode absolute`.
- Mixing the two adds the state twice, and the arms freeze.

**Leave `max_token_len` at the π0.5 default.** π0.5 reads proprioception only as
discretised state text in the prompt. A shorter limit truncates the right arm and
hand out of it. The tokenizer only logs a warning when this happens.

**Prompts must match the training task strings.** Use them word for word, for
example `pick up the mango and place it in the basket`. Other wording degrades
the policy.

**Compute norm stats with the G1 script.**
- `compute_norm_stats_g1_dex3.py` reads only the parquet files.
- It falls back to min/max for joints that barely move (the left hand), where
  quantile normalisation would blow up.
