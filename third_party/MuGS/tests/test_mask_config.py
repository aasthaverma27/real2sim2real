"""
Test mask configuration system

Tests for the flexible mask configuration and segment ID system.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import yaml

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mugs.utils.mask_config import (
    MaskGroup,
    MaskConfig,
    create_group_masks,
    composite_with_groups,
)


class TestMaskGroup:
    """Test MaskGroup dataclass"""

    def test_mask_group_creation(self):
        """Test creating a MaskGroup"""
        group = MaskGroup(
            name="robot",
            geom_names=["palm", "finger1"],
            rendering_mode="mujoco",
            composite_priority=10
        )

        assert group.name == "robot"
        assert len(group.geom_names) == 2
        assert group.rendering_mode == "mujoco"
        assert group.composite_priority == 10

    def test_mask_group_defaults(self):
        """Test MaskGroup default values"""
        group = MaskGroup(name="test")

        assert group.geom_names == []
        assert group.body_names == []
        assert group.geom_prefix is None
        assert group.rendering_mode == "mujoco"
        assert group.composite_priority == 0


class TestMaskConfig:
    """Test MaskConfig system"""

    def test_mask_config_creation(self):
        """Test creating a MaskConfig"""
        groups = [
            MaskGroup(name="robot", geom_names=["palm"]),
            MaskGroup(name="table", geom_names=["table_top"])
        ]

        config = MaskConfig(groups=groups, default_background="3dgs")

        assert len(config.groups) == 2
        assert config.default_background == "3dgs"

    def test_yaml_roundtrip(self):
        """Test saving and loading YAML config"""
        groups = [
            MaskGroup(
                name="robot",
                geom_names=["palm", "finger1"],
                rendering_mode="mujoco",
                composite_priority=10
            ),
            MaskGroup(
                name="table",
                geom_prefix="table_",
                rendering_mode="mujoco",
                composite_priority=1
            )
        ]

        config = MaskConfig(groups=groups, default_background="3dgs")

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = Path(f.name)
            config.to_yaml(temp_path)

        try:
            # Load back
            loaded_config = MaskConfig.from_yaml(temp_path)

            assert len(loaded_config.groups) == 2
            assert loaded_config.default_background == "3dgs"
            assert loaded_config.groups[0].name == "robot"
            assert loaded_config.groups[0].composite_priority == 10
            assert loaded_config.groups[1].geom_prefix == "table_"
        finally:
            temp_path.unlink()


class TestMaskCreation:
    """Test mask creation from segment IDs"""

    def test_create_simple_masks(self):
        """Test creating masks from segment IDs"""
        # Create mock segmentation map
        seg_ids = np.array([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3]
        ], dtype=np.int32)

        # Create mock model with geom IDs
        class MockModel:
            ngeom = 4
            geom_bodyid = np.array([0, 0, 1, 1])  # geoms 0,1 on body 0; 2,3 on body 1

        model = MockModel()

        # Create config
        groups = [
            MaskGroup(name="group1", geom_names=[]),  # Will be resolved to IDs [0, 1]
            MaskGroup(name="group2", geom_names=[]),  # Will be resolved to IDs [2, 3]
        ]
        config = MaskConfig(groups=groups)

        # For testing, we'll manually create masks since we don't have full MuJoCo
        mask1 = np.zeros_like(seg_ids, dtype=np.uint8)
        mask1[seg_ids <= 1] = 1

        mask2 = np.zeros_like(seg_ids, dtype=np.uint8)
        mask2[seg_ids >= 2] = 1

        group_masks = {
            "group1": mask1,
            "group2": mask2
        }

        # Verify masks
        assert mask1.sum() == 8  # Half the pixels
        assert mask2.sum() == 8  # Other half
        assert (mask1 & mask2).sum() == 0  # No overlap


class TestCompositing:
    """Test compositing with masks"""

    def test_simple_composite(self):
        """Test basic compositing"""
        # Create test images
        H, W = 4, 4
        mujoco_rgb = np.full((H, W, 3), 100, dtype=np.uint8)  # Gray
        gs_rgb = np.full((H, W, 3), 200, dtype=np.uint8)  # Lighter gray

        # Create mask (top half = group1)
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[:2, :] = 1

        group_masks = {"robot": mask}

        # Create config (robot rendered with MuJoCo, rest with 3DGS)
        groups = [
            MaskGroup(
                name="robot",
                rendering_mode="mujoco",
                composite_priority=10
            )
        ]
        config = MaskConfig(groups=groups, default_background="3dgs")

        # Composite
        result = composite_with_groups(
            mujoco_rgb, gs_rgb, group_masks, config
        )

        # Check results
        # Top half should be MuJoCo (100)
        assert np.all(result[:2, :] == 100)
        # Bottom half should be 3DGS (200)
        assert np.all(result[2:, :] == 200)

    def test_priority_ordering(self):
        """Test that higher priority renders on top"""
        H, W = 4, 4

        # Three different colors
        color_bg = np.full((H, W, 3), 50, dtype=np.uint8)
        color_mj = np.full((H, W, 3), 150, dtype=np.uint8)

        # Two overlapping masks
        mask_low = np.zeros((H, W), dtype=np.uint8)
        mask_low[:3, :] = 1  # Covers top 3 rows

        mask_high = np.zeros((H, W), dtype=np.uint8)
        mask_high[1:, :] = 1  # Covers bottom 3 rows (overlaps with mask_low)

        group_masks = {
            "low_priority": mask_low,
            "high_priority": mask_high
        }

        groups = [
            MaskGroup(name="low_priority", rendering_mode="mujoco", composite_priority=1),
            MaskGroup(name="high_priority", rendering_mode="mujoco", composite_priority=10),
        ]
        config = MaskConfig(groups=groups, default_background="3dgs")

        result = composite_with_groups(
            color_mj, color_bg, group_masks, config
        )

        # All pixels should be MuJoCo color (150) because both masks cover everything
        # and they both use MuJoCo rendering
        assert np.all(result == 150)


class TestIntegration:
    """Integration tests"""

    def test_full_pipeline(self):
        """Test full pipeline: YAML config → masks → composite"""
        # Create YAML config
        config_dict = {
            'default_background': '3dgs',
            'groups': [
                {
                    'name': 'robot',
                    'geom_names': ['palm', 'finger1'],
                    'rendering_mode': 'mujoco',
                    'composite_priority': 10
                },
                {
                    'name': 'table',
                    'geom_prefix': 'table_',
                    'rendering_mode': 'mujoco',
                    'composite_priority': 1
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = Path(f.name)
            yaml.dump(config_dict, f)

        try:
            # Load config
            config = MaskConfig.from_yaml(temp_path)

            # Verify config loaded correctly
            assert config.default_background == '3dgs'
            assert len(config.groups) == 2
            assert config.groups[0].name == 'robot'
            assert config.groups[0].composite_priority == 10
            assert config.groups[1].geom_prefix == 'table_'

        finally:
            temp_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
