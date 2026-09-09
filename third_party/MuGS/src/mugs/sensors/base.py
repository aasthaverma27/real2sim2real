"""
Base sensor interface for MuGS.

Provides standalone sensor base class for photorealistic rendering.
For mjlab integration, see the mugs_mjlab package.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class SensorBase(ABC):
    """
    Base class for MuGS standalone sensors.

    All sensors must implement the render() method and expose width/height properties.
    """

    @abstractmethod
    def render(self, *args, **kwargs) -> np.ndarray:
        """
        Render sensor observation.

        Returns:
            RGB image as numpy array (H, W, 3) uint8
        """
        pass

    def get_observation_space(self) -> Dict[str, Any]:
        """
        Get observation space specification.

        Returns:
            Dictionary with shape, dtype, and bounds
        """
        return {
            "shape": (self.height, self.width, 3),
            "dtype": np.uint8,
            "low": 0,
            "high": 255,
        }

    @property
    @abstractmethod
    def width(self) -> int:
        """Sensor width in pixels."""
        pass

    @property
    @abstractmethod
    def height(self) -> int:
        """Sensor height in pixels."""
        pass

    def get_info(self) -> Dict[str, Any]:
        """Get sensor metadata and statistics."""
        return {
            "type": self.__class__.__name__,
            "resolution": (self.width, self.height),
        }
