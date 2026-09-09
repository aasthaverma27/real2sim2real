"""
MuGS-MJLab Integration

Provides mjlab-compatible sensors for hybrid MuJoCo + 3DGS rendering.
This package requires both mugs and mjlab to be installed.

Example:
    ```python
    from mugs_mjlab.sensors import GaussianSensorMjlab, GaussianSensorMjlabCfg
    from mjlab import Environment

    cfg = GaussianSensorMjlabCfg(
        name="camera1",
        resolution=(640, 480),
        scene_ply="kitchen.ply"
    )

    env = Environment(sensors=[cfg])
    ```
"""

__version__ = "0.1.0"

from mugs_mjlab.sensors import GaussianSensorMjlab, GaussianSensorMjlabCfg
from mugs_mjlab.tasks import (
    BenchmarkResult,
    benchmark_env_cfg,
    kitchen_ply_path,
    make_yam_lift_cube_gs_env_cfg,
    print_benchmark_table,
)

__all__ = [
    "__version__",
    "GaussianSensorMjlab",
    "GaussianSensorMjlabCfg",
    "make_yam_lift_cube_gs_env_cfg",
    "kitchen_ply_path",
    "benchmark_env_cfg",
    "print_benchmark_table",
    "BenchmarkResult",
]
