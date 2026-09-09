"""
Pytest configuration and fixtures

Common fixtures for MuGS tests.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_seg_ids():
    """Sample segmentation ID map for testing"""
    return np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3]
    ], dtype=np.int32)


@pytest.fixture
def sample_rgb_images():
    """Sample RGB images for compositing tests"""
    H, W = 4, 4
    mujoco_rgb = np.full((H, W, 3), 100, dtype=np.float32)
    gs_rgb = np.full((H, W, 3), 200, dtype=np.float32)
    return mujoco_rgb, gs_rgb


@pytest.fixture
def sample_mask():
    """Sample binary mask"""
    mask = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1]
    ], dtype=np.uint8)
    return mask


@pytest.fixture
def temp_config_path(tmp_path):
    """Temporary path for config files"""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    return config_dir
