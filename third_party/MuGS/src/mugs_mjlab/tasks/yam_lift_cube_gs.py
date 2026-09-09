"""YAM lift-cube task with a `GaussianSensorMjlab` attached.

Thin wrapper around `mjlab.tasks.manipulation.config.yam.yam_lift_cube_env_cfg`
that pre-attaches a wrist-mounted hybrid (3DGS + MuJoCo) Gaussian sensor so
end-users get a photorealistic manipulation env in two lines.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mjlab.tasks.manipulation.config.yam import yam_lift_cube_env_cfg

from mugs_mjlab.sensors import GaussianSensorMjlabCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnvCfg
    from mugs_mjlab.sensors.gaussian_sensor import RenderMode


# Pretrained kitchen scene shipped with this repo.
_KITCHEN_PLY_REL = Path("data/pretrained/kitchen/point_cloud/iteration_30000/point_cloud.ply")


def kitchen_ply_path() -> Path:
    """Locate the pretrained kitchen-scene PLY checked into the MuGS repo.

    Walks parents of this file looking for the asset, so it works in editable
    installs regardless of where the repo lives on disk.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / _KITCHEN_PLY_REL
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Pretrained kitchen PLY not found at any parent of {here}. "
        f"Looked for: */{_KITCHEN_PLY_REL}"
    )


def make_yam_lift_cube_gs_env_cfg(
    ply_path: str | Path,
    *,
    num_envs: int = 64,
    width: int = 160,
    height: int = 120,
    camera_name: str = "robot/camera_d405",
    sensor_name: str = "gs_rgb",
    render_mode: "RenderMode" = "hybrid",
    play: bool = True,
) -> "ManagerBasedRlEnvCfg":
    """Build a YAM lift-cube env config with a Gaussian sensor attached.

    Args:
        ply_path: Path to the 3DGS PLY background.
        num_envs: Number of parallel environments.
        width, height: Sensor image resolution.
        camera_name: MuJoCo camera the sensor wraps (must already exist in
            the scene — defaults to the YAM robot's wrist camera).
        sensor_name: Sensor key under which mjlab exposes its output.
        render_mode: "hybrid" composites 3DGS background with MuJoCo
            foreground; "3dgs_only" / "mujoco_only" skip the other branch.
        play: Passed through to `yam_lift_cube_env_cfg` (disables curriculum
            and obs-corruption when True).

    Returns:
        A `ManagerBasedRlEnvCfg` ready to pass to `ManagerBasedRlEnv`.
    """
    cfg = yam_lift_cube_env_cfg(play=play)
    cfg.scene.num_envs = num_envs

    sensor_cfg = GaussianSensorMjlabCfg(
        name=sensor_name,
        camera_name=camera_name,
        width=width,
        height=height,
        background_ply_path=Path(ply_path),
        render_mode=render_mode,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (sensor_cfg,)
    return cfg
