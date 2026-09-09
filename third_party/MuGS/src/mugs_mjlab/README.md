# MuGS-MJLab Integration

MJLab-compatible sensors for photorealistic robot simulation using 3D Gaussian Splatting.

## Installation

```bash
# Install mugs first
pip install -e .

# Install mugs_mjlab (requires mjlab)
pip install -e . -c pyproject_mugs_mjlab.toml
```

## Usage

```python
from mugs_mjlab.sensors import GaussianSensorMjlab, GaussianSensorMjlabCfg
from mjlab import Environment

# Configure sensor
cfg = GaussianSensorMjlabCfg(
    name="photorealistic_camera",
    width=640,
    height=480,
    camera_name="robot/head_cam",
    background_ply_path="scenes/kitchen.ply",
    render_mode="hybrid"
)

# Use in mjlab environment
env = Environment(
    model_path="robot.xml",
    sensors=[cfg]
)

# Sensor renders automatically
obs = env.reset()
rgb = obs['photorealistic_camera']  # (num_envs, H, W, 3) torch.Tensor
```

## Architecture

This package provides a thin integration layer between:
- **mugs**: Core standalone sensors and rendering utilities
- **mjlab**: Batched physics simulation framework

The `GaussianSensorMjlab` class inherits from `mjlab.sensor.Sensor` and uses `mugs` utilities for the actual rendering pipeline.

## Dependencies

- `mugs>=0.1.0` - Core MuGS package (standalone)
- `mjlab>=0.1.0` - MJLab simulation framework
- `torch>=2.0.0`
- `gsplat>=1.5.0`

## See Also

- [mugs](../mugs/) - Standalone sensors (no mjlab dependency)
- [MuGS Documentation](../../docs/)
