# DISCOVERSE Full Demo

Complete demonstration of using DISCOVERSE assets with MuGS.

## What This Demo Does

1. **Auto-setup**: Clones DISCOVERSE repository if not present
2. **Asset Discovery**: Lists all available tasks and 3DGS scenes
3. **Task Execution**: Runs DISCOVERSE manipulation tasks
4. **Hybrid Rendering**: Renders with MuGS (3DGS + MuJoCo)
5. **Visualization**: Generates detailed comparison images

## Quick Start

```bash
# Run the complete demo (will auto-setup DISCOVERSE)
TORCH_CUDA_ARCH_LIST="8.6" python examples/discoverse_full_demo.py
```

The script will:
- Clone DISCOVERSE automatically
- Discover available tasks and 3DGS scenes
- Run 3 demo tasks
- Generate comparison visualizations

## Output

Demo generates images in `outputs/discoverse_demo/`:
- `{task_name}_demo.jpg` - 8-panel comparison for each task
- Shows: MuJoCo Only, 3DGS Only, MuGS Hybrid, Mask, Details, Statistics

## Available DISCOVERSE Tasks

### Manipulation Tasks
- `stack_block` - Stack building blocks
- `pan_pick` - Pick up cooking pan
- `place_coffeecup` - Place coffee cup
- `kiwi_pick` - Pick kiwi fruit
- `block_bridge_place` - Bridge construction
- `close_laptop` - Close laptop lid
- `cover_cup` - Cover cup with lid
- `open_drawer` - Open drawer
- `peg_in_hole` - Peg insertion
- `pick_jujube` - Pick jujube fruit
- `push_mouse` - Push computer mouse

### Robots
- **AIRBOT Play**: Lightweight 6-DOF arm
- **AIRBOT MMK2**: Dual-arm humanoid mobile manipulator

## What You'll See

### Demo Output Example

Each task generates an 8-panel visualization:

**Row 1: Main Comparison**
1. MuJoCo Only (Physics Simulation)
2. 3DGS Only (Photorealistic Background)
3. MuGS Hybrid (Combined Result) ← **Key output**
4. Robot Mask (Blending weights)

**Row 2: Details**
5. MuJoCo Detail (Cropped robot view)
6. Robot Highlight (Red overlay on background)
7. Blend Weights (Heat map)
8. Statistics Panel (Task info, coverage, etc.)

## Manual Setup (if auto-setup fails)

```bash
# Clone DISCOVERSE manually
git clone https://github.com/TATP-233/DISCOVERSE data/external/DISCOVERSE
cd data/external/DISCOVERSE

# Pull LFS assets (optional, will auto-download otherwise)
git lfs install
git lfs pull

# Return to project root
cd ../../..

# Run demo
python examples/discoverse_full_demo.py
```

## Run Specific Task

Modify `discoverse_full_demo.py` to run a specific task:

```python
# At the end of main(), add:
run_demo_task(
    task_path=Path("data/external/DISCOVERSE/models/mjcf/task_environments/stack_block.xml"),
    scene_ply=Path("data/external/DISCOVERSE/models/3dgs/scene/kitchen.ply"),
)
```

## Troubleshooting

### No tasks found
```
Problem: No MJCF files in models/mjcf/
Solution: DISCOVERSE repo may be incomplete. Try:
  cd data/external/DISCOVERSE && git pull
```

### No 3DGS scenes
```
Problem: models/3dgs/scene/ is empty
Solution: 3DGS files auto-download on first DISCOVERSE run.
  Run a DISCOVERSE example first:
  cd data/external/DISCOVERSE
  python examples/tasks_airbot_play/place_coffeecup.py
```

### CUDA compilation error
```
Problem: nvcc fatal: Unsupported gpu architecture
Solution: Set TORCH_CUDA_ARCH_LIST:
  TORCH_CUDA_ARCH_LIST="8.6" python examples/discoverse_full_demo.py
```

## Features

### Auto-Setup
- Automatically clones DISCOVERSE if not present
- Checks for required directories
- Attempts LFS pull for large assets

### Asset Management
- Lists all available tasks
- Finds 3DGS scene files
- Fallback to pretrained kitchen scene
- Automatic geom extraction from MJCF

### Rendering
- MuGS hybrid mode (3DGS + MuJoCo)
- Robot segmentation mask
- Alpha compositing
- Camera parameter extraction

### Visualization
- 8-panel comparison layout
- Robot detail cropping
- Statistical overlays
- High-resolution output (150 DPI)

## Performance

- **Setup Time**: ~1 min (first run, cloning repo)
- **Render Time**: ~2 seconds per task (after 3DGS loading)
- **3DGS Loading**: ~2 seconds (cached after first use)
- **FPS**: ~5000 (640×480 resolution)

## Comparison: DISCOVERSE vs GS-Playground

| Feature | DISCOVERSE | GS-Playground |
|---------|------------|---------------|
| **Tasks** | 12+ manipulation | Locomotion + manipulation |
| **3DGS** | Auto-download | Manual + Hugging Face |
| **Robots** | AIRBOT Play, MMK2 | Quadrupeds, Humanoids, Arms |
| **Integration** | ROS, MJCF, URDF | MuJoCo parallel physics |
| **License** | MIT | Check repo |

## Next Steps

1. **View Results**: Check `outputs/discoverse_demo/`
2. **Try More Tasks**: Modify demo to run all 12+ tasks
3. **Custom Scenes**: Create your own MJCF with DISCOVERSE backgrounds
4. **Benchmark**: Compare rendering performance
5. **Sim2Real**: Use for policy training and transfer

## Resources

- **DISCOVERSE GitHub**: https://github.com/TATP-233/DISCOVERSE
- **DISCOVERSE Paper**: https://arxiv.org/abs/2507.21981
- **MuGS Docs**: `docs/DISCOVERSE_INTEGRATION.md`
- **Integration Guide**: `docs/EXTERNAL_ASSETS.md`

## Citation

If you use DISCOVERSE assets in your research:

```bibtex
@inproceedings{discoverse2025,
  title={DISCOVERSE: Efficient Robot Simulation in Complex High-Fidelity Environments},
  author={Jia, Yufei and others},
  booktitle={IROS},
  year={2025}
}
```

For MuGS:

```bibtex
@article{mugs2026,
  title={MuGS: Photorealistic Simulation for Vision-Language-Action Models},
  journal={RSS},
  year={2026}
}
```
