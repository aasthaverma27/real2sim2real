"""
Mask Configuration System

Flexible configuration for selecting which MuJoCo geoms to mask/composite.
Supports grouping by object type, body, or custom selection.

Author: MuGS Team
Date: 2026-05-02
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import mujoco
import numpy as np
import yaml


@dataclass
class MaskGroup:
    """Configuration for a group of geoms to mask together."""

    name: str
    """Group name (e.g., 'robot', 'objects', 'table')"""

    geom_names: List[str] = field(default_factory=list)
    """Explicit list of geom names"""

    body_names: List[str] = field(default_factory=list)
    """Body names - includes all geoms under these bodies"""

    geom_prefix: Optional[str] = None
    """Geom name prefix pattern (e.g., 'robot_')"""

    rendering_mode: str = "mujoco"
    """How to render this group: 'mujoco' or '3dgs' or 'both'"""

    composite_priority: int = 0
    """Compositing order (higher = rendered on top)"""


@dataclass
class MaskConfig:
    """Complete mask configuration for hybrid rendering."""

    groups: List[MaskGroup] = field(default_factory=list)
    """List of mask groups"""

    default_background: str = "3dgs"
    """Default rendering for unmasked areas: 'mujoco' or '3dgs'"""

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "MaskConfig":
        """Load configuration from YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        groups = [MaskGroup(**group_data) for group_data in data.get("groups", [])]
        return cls(groups=groups, default_background=data.get("default_background", "3dgs"))

    def to_yaml(self, yaml_path: Path):
        """Save configuration to YAML file."""
        data = {
            "default_background": self.default_background,
            "groups": [
                {
                    "name": g.name,
                    "geom_names": g.geom_names,
                    "body_names": g.body_names,
                    "geom_prefix": g.geom_prefix,
                    "rendering_mode": g.rendering_mode,
                    "composite_priority": g.composite_priority,
                }
                for g in self.groups
            ],
        }

        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def resolve_geom_ids(model: mujoco.MjModel, mask_group: MaskGroup) -> List[int]:
    """
    Resolve mask group configuration to actual geom IDs.

    Args:
        model: MuJoCo model
        mask_group: Mask group configuration

    Returns:
        List of geom IDs
    """
    geom_ids = []

    # 1. Explicit geom names
    for geom_name in mask_group.geom_names:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id >= 0:
            geom_ids.append(geom_id)

    # 2. Geoms under specified bodies
    for body_name in mask_group.body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        if body_id >= 0:
            # Find all geoms belonging to this body
            for geom_id in range(model.ngeom):
                if model.geom_bodyid[geom_id] == body_id:
                    geom_ids.append(geom_id)

    # 3. Geoms matching prefix
    if mask_group.geom_prefix:
        for geom_id in range(model.ngeom):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name and geom_name.startswith(mask_group.geom_prefix):
                geom_ids.append(geom_id)

    # Remove duplicates and sort
    return sorted(set(geom_ids))


def create_group_masks(
    seg_ids: "np.ndarray", model: mujoco.MjModel, mask_config: MaskConfig
) -> Dict[str, "np.ndarray"]:
    """
    Create masks for all configured groups.

    Args:
        seg_ids: Segmentation IDs (H, W)
        model: MuJoCo model
        mask_config: Mask configuration

    Returns:
        Dictionary mapping group name to binary mask (H, W)
    """
    import numpy as np

    masks = {}

    for group in mask_config.groups:
        geom_ids = resolve_geom_ids(model, group)

        # Create binary mask
        mask = np.zeros_like(seg_ids, dtype=np.uint8)
        for geom_id in geom_ids:
            mask[seg_ids == geom_id] = 1

        masks[group.name] = mask

    return masks


def composite_with_groups(
    mujoco_rgb: "np.ndarray",
    gs_rgb: "np.ndarray",
    group_masks: Dict[str, "np.ndarray"],
    mask_config: MaskConfig,
) -> "np.ndarray":
    """
    Composite images using group masks and configuration.

    Args:
        mujoco_rgb: MuJoCo RGB (H, W, 3)
        gs_rgb: 3DGS RGB (H, W, 3)
        group_masks: Dictionary of group masks
        mask_config: Mask configuration

    Returns:
        Composited RGB image (H, W, 3)
    """
    import numpy as np

    # Start with default background
    if mask_config.default_background == "3dgs":
        composite = gs_rgb.copy()
    else:
        composite = mujoco_rgb.copy()

    # Sort groups by priority (lowest first, so highest rendered on top)
    sorted_groups = sorted(mask_config.groups, key=lambda g: g.composite_priority)

    for group in sorted_groups:
        if group.name not in group_masks:
            continue

        mask = group_masks[group.name]
        mask_3ch = mask[..., None].astype(np.float32)

        # Select source based on rendering mode
        if group.rendering_mode == "mujoco":
            source = mujoco_rgb
        elif group.rendering_mode == "3dgs":
            source = gs_rgb
        elif group.rendering_mode == "both":
            # Average of both (experimental)
            source = (mujoco_rgb + gs_rgb) / 2
        else:
            continue

        # Composite this group
        composite = composite * (1 - mask_3ch) + source * mask_3ch

    return composite


# Example configurations


def create_default_robot_config() -> MaskConfig:
    """Create default configuration for robot manipulation."""
    return MaskConfig(
        groups=[
            MaskGroup(
                name="robot",
                body_names=["robot_base", "robot_arm"],  # All geoms under these bodies
                rendering_mode="mujoco",
                composite_priority=10,  # Rendered on top
            ),
            MaskGroup(
                name="gripper",
                geom_prefix="finger",  # All geoms starting with 'finger'
                rendering_mode="mujoco",
                composite_priority=11,  # Even higher priority
            ),
            MaskGroup(
                name="table",
                geom_names=["table_top", "table_leg1", "table_leg2", "table_leg3", "table_leg4"],
                rendering_mode="mujoco",
                composite_priority=5,
            ),
        ],
        default_background="3dgs",  # Everything else rendered with 3DGS
    )


def create_kitchen_config() -> MaskConfig:
    """Create configuration for kitchen scene."""
    return MaskConfig(
        groups=[
            MaskGroup(
                name="robot",
                geom_names=[
                    "palm",
                    "finger1_link",
                    "finger2_link",
                    "finger3_link",
                    "finger4_link",
                    "finger5_link",
                ],
                rendering_mode="mujoco",
                composite_priority=10,
            ),
            MaskGroup(
                name="table", geom_prefix="table_", rendering_mode="mujoco", composite_priority=1
            ),
        ],
        default_background="3dgs",
    )
