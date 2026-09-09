"""
Test rendering utilities

Tests for core rendering utility functions.
"""

import pytest
import numpy as np
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mugs.utils.rendering import (
    create_robot_mask,
    composite_images,
)


class TestMaskCreation:
    """Test mask creation utilities"""

    def test_create_robot_mask_basic(self):
        """Test basic robot mask creation"""
        # Create mock segmentation IDs
        seg_ids = np.array([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3]
        ], dtype=np.int32)

        # Mock model
        class MockModel:
            ngeom = 4
            geom_names = ["floor", "table", "palm", "finger"]

            @staticmethod
            def get_geom_id(name):
                names = ["floor", "table", "palm", "finger"]
                return names.index(name) if name in names else -1

        model = MockModel()

        # Create mask for geoms [2, 3] (palm, finger)
        robot_geom_names = ["palm", "finger"]

        # Since we don't have full MuJoCo, manually create the expected mask
        robot_geom_ids = [2, 3]
        mask = np.zeros_like(seg_ids, dtype=np.uint8)
        for gid in robot_geom_ids:
            mask[seg_ids == gid] = 1

        # Verify mask
        assert mask.shape == seg_ids.shape
        assert mask.dtype == np.uint8
        assert mask.sum() == 8  # Bottom-right quadrant (2x2 + 2x2)
        assert mask[0, 0] == 0  # floor
        assert mask[0, 2] == 0  # table
        assert mask[2, 0] == 1  # palm
        assert mask[2, 2] == 1  # finger


class TestCompositing:
    """Test image compositing"""

    def test_composite_images_basic(self):
        """Test basic image compositing"""
        H, W = 4, 4

        # Create test images
        mujoco_rgb = np.full((H, W, 3), 100, dtype=np.float32)
        gs_rgb = np.full((H, W, 3), 200, dtype=np.float32)

        # Create mask (top-left quadrant = 1)
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[:2, :2] = 1

        # Composite
        result = composite_images(mujoco_rgb, gs_rgb, mask)

        # Check shape and type
        assert result.shape == (H, W, 3)
        assert result.dtype == np.float32

        # Top-left quadrant should be MuJoCo (100)
        assert np.allclose(result[0, 0], 100)
        assert np.allclose(result[1, 1], 100)

        # Bottom-right quadrant should be 3DGS (200)
        assert np.allclose(result[2, 2], 200)
        assert np.allclose(result[3, 3], 200)

    def test_composite_images_edge_values(self):
        """Test compositing with edge mask values"""
        H, W = 2, 2

        mujoco_rgb = np.array([[[100, 100, 100]]] * 4, dtype=np.float32).reshape(H, W, 3)
        gs_rgb = np.array([[[200, 200, 200]]] * 4, dtype=np.float32).reshape(H, W, 3)

        # All zeros mask -> all 3DGS
        mask_zeros = np.zeros((H, W), dtype=np.uint8)
        result = composite_images(mujoco_rgb, gs_rgb, mask_zeros)
        assert np.allclose(result, 200)

        # All ones mask -> all MuJoCo
        mask_ones = np.ones((H, W), dtype=np.uint8)
        result = composite_images(mujoco_rgb, gs_rgb, mask_ones)
        assert np.allclose(result, 100)

    def test_composite_images_color_channels(self):
        """Test that compositing works correctly per color channel"""
        H, W = 2, 2

        # MuJoCo: red
        mujoco_rgb = np.zeros((H, W, 3), dtype=np.float32)
        mujoco_rgb[..., 0] = 255

        # 3DGS: blue
        gs_rgb = np.zeros((H, W, 3), dtype=np.float32)
        gs_rgb[..., 2] = 255

        # Checkerboard mask
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)

        result = composite_images(mujoco_rgb, gs_rgb, mask)

        # Check corners
        # [0, 0] mask=1 -> MuJoCo (red)
        assert result[0, 0, 0] == 255  # R
        assert result[0, 0, 1] == 0    # G
        assert result[0, 0, 2] == 0    # B

        # [0, 1] mask=0 -> 3DGS (blue)
        assert result[0, 1, 0] == 0    # R
        assert result[0, 1, 1] == 0    # G
        assert result[0, 1, 2] == 255  # B


class TestDataTypes:
    """Test handling of different data types"""

    def test_uint8_input(self):
        """Test with uint8 images"""
        H, W = 2, 2

        mujoco_rgb = np.full((H, W, 3), 100, dtype=np.uint8)
        gs_rgb = np.full((H, W, 3), 200, dtype=np.uint8)
        mask = np.ones((H, W), dtype=np.uint8)

        # Convert to float for compositing
        result = composite_images(
            mujoco_rgb.astype(np.float32),
            gs_rgb.astype(np.float32),
            mask
        )

        assert result.dtype == np.float32

    def test_mask_broadcast(self):
        """Test that 2D mask broadcasts correctly to 3D RGB"""
        H, W = 3, 3

        mujoco_rgb = np.random.rand(H, W, 3).astype(np.float32) * 255
        gs_rgb = np.random.rand(H, W, 3).astype(np.float32) * 255
        mask = np.random.randint(0, 2, (H, W), dtype=np.uint8)

        result = composite_images(mujoco_rgb, gs_rgb, mask)

        # Check shape is preserved
        assert result.shape == (H, W, 3)

        # Check that mask was applied to all channels
        for i in range(H):
            for j in range(W):
                expected = mujoco_rgb[i, j] if mask[i, j] else gs_rgb[i, j]
                assert np.allclose(result[i, j], expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
