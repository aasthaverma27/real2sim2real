# External dependencies, and why they are submodules

Nothing external is checked into this repository. Every external stack is a
**submodule** — a URL and a commit SHA — so this repo distributes none of that
code and inherits none of its licence terms.

That is a licence decision, not a style preference. Please do not "simplify" it
by vendoring copies.

## What each one is

| submodule | licence | public? | needed for |
|---|---|---|---|
| `external/Isaac-GR00T` | Apache-2.0 | yes | the VLA policy server |
| `external/unitree_sdk2_python` | Unitree, permissive | yes | the DDS bridge |
| `external/simple_wbc_code` | none stated | third-party | balance/walk ONNX, keyboard teleop |
| `external/vla_for_simulation_testing` | mixed — see below | **no** | DDS sim bridge, WBC container, VLA client |

Each is pinned to the commit this stack was actually tested against, not to a
moving branch:

```
Isaac-GR00T                 b9955401
unitree_sdk2_python         a035adea
vla_for_simulation_testing  ce025211
simple_wbc_code             6c0d5101
```

## The constraint that shapes all of this

`vla_for_simulation_testing/deploy-wbc-on-robot` — the whole-body controller —
is under the **NVIDIA License**. Two clauses matter:

* **§3.3 Use Limitation.** The work and derivatives "only may be used or
  intended for use non-commercially… for non-commercial research purposes
  only", explicitly excluding military, surveillance, nuclear and biometric
  purposes.
* **§3.1 Redistribution.** You may distribute it only under that same licence,
  including a complete copy of it, and retaining every copyright, patent,
  trademark and attribution notice unmodified.

So vendoring the controller into this repository would carry those terms into
the whole distribution. A submodule does not: it points at code the user
obtains directly from its licensor, under whatever terms apply to them.

The same reasoning covers the **ONNX policy weights**
(`GR00T-WholeBodyControl-Balance.onnx`, `…-Walk.onnx`). They are NVIDIA GR00T
whole-body-control artefacts, and `.onnx` is in `.gitignore` so they cannot be
committed by accident. Point `RS_WBC_MODELS` at wherever you obtained them.

`gear_sonic`, in the same private repository, carries **no licence file at
all** — which means no redistribution rights are granted, so it must stay a
submodule regardless.

## Getting them

```bash
scripts/setup_external.sh --check     # what is present, what is missing
scripts/setup_external.sh             # fetch what you can reach
```

Then record the locations once, in `configs/paths.env`:

```bash
cp configs/paths.env.example configs/paths.env
$EDITOR configs/paths.env
```

## If you cannot access the private repository

Most of this project still works. With **no external stack at all** you get
reconstruction, the metric MuJoCo scene, the photoreal hybrid render, the robot
cameras, and a working whole-body controller running in-process:

```bash
scripts/realsim/start.sh --builtin
```

That path needs only the two `.onnx` files.

What you lose is the **deployed** controller over DDS and the VLA client that
feeds it — `scripts/run_mujoco_wbc_sim.py`, `scripts/wbc_state_relay.py` and
the four-process stack in [pipeline_commands.md](pipeline_commands.md). Those
need `gear_sonic` for the DDS bridge and the container for the controller.
There is no public substitute; ask whoever owns your control stack.

## Weights are never in git

VLA checkpoints run to tens of GB. They are not submodules and not committed.
Fetch them separately and set `RS_GROOT_CHECKPOINT`. Reconstruction model
weights come from `python scripts/download_auto_checkpoints.py` (~134 GB for
the full set).
