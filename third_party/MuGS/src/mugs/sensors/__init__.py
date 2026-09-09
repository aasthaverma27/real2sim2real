"""
MuGS Sensors: Photorealistic rendering sensors for MuJoCo.

Standalone sensors only. For mjlab integration, see mugs_mjlab package.
"""

from .base import SensorBase
from .gaussian_sensor import GaussianSensor, GaussianSensorConfig

__all__ = [
    "SensorBase",
    "GaussianSensor",
    "GaussianSensorConfig",
]
