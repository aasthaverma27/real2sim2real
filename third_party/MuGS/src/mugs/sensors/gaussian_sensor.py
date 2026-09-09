"""
GaussianSensor: 3DGS-based sensor for photorealistic rendering

Implements hybrid rendering with:
- 3DGS background (static or pretrained scenes)
- MuJoCo foreground (physics-simulated objects/robots)
- Segmentation-based compositing

Standalone sensor for use without mjlab.
For mjlab integration, use mugs_mjlab.sensors.GaussianSensorMjlab instead.

Author: MuGS Team
Date: 2026-05-02
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import mujoco
import numpy as np
import torch
from plyfile import PlyData

from .base import SensorBase

try:
    from gsplat import rasterization

    GSPLAT_AVAILABLE = True
except ImportError:
    GSPLAT_AVAILABLE = False
    print("Warning: gsplat not available, GaussianSensor will not work")


@dataclass
class GaussianSensorConfig:
    """Configuration for GaussianSensor."""

    # Resolution
    width: int = 640
    height: int = 480

    # Camera parameters
    fov_degrees: float = 60.0
    camera_name: str = "main_camera"

    # Background scene
    background_ply_path: Optional[Path] = None
    """Path to 3DGS background scene PLY file"""

    # Rendering mode
    render_mode: str = "hybrid"
    """'hybrid', '3dgs_only', or 'mujoco_only'"""

    # Robot masking
    robot_geom_names: List[str] = field(
        default_factory=lambda: [
            "base_link",
            "shoulder_link",
            "arm_link",
            "forearm_link",
            "palm",
            "left_finger_link",
            "right_finger_link",
        ]
    )
    """MuJoCo geom names to extract as robot mask"""

    robot_geom_ids: Optional[List[int]] = None

    splat_geom_ids: Optional[List[int]] = None
    """Geoms whose pixels are painted with the splat's colour instead of the
    mesh's (Re3Sim's rule: mix_mask = table_mask | ground_mask, then
    np.where(mix_mask, gaussians, sim)). The mesh still defines the silhouette
    and the depth, so a table treated this way can never appear twice however
    badly the splat is aligned - there is no second silhouette to disagree with.
    Where the splat has no coverage the mesh's own render is kept, so a novel
    viewpoint the capture never saw degrades to the mesh instead of to black."""
    """MuJoCo geom IDs to extract as robot mask (alternative to names)"""

    # Performance
    device: str = "cuda"
    cache_background: bool = True
    """Cache static 3DGS background for speed"""

    # Super-resolution (future)
    enable_sr: bool = False
    sr_model_path: Optional[Path] = None

    # ── RoboSnap fork additions (see FORK_NOTES.md) ──────────────────
    use_sh: bool = True
    """Render with the PLY's full spherical harmonics when present. Upstream
    keeps only the DC term, which flattens view-dependent shading."""

    depth_composite: bool = False
    """Composite by depth test rather than a binary mask, so foreground geometry
    that is behind the splat stays hidden."""

    depth_tolerance: float = 1.02
    """Foreground wins where fg_depth <= bg_depth * depth_tolerance."""

    disable_rangefinder: bool = True
    """Hide rangefinder rays; the G1's 512 lidar rays otherwise swamp the frame."""

    camera_convention: str = "mujoco"
    """Convention of camera_params['rotation_matrix']. 'mujoco' (x right, y up,
    z backward) needs BOTH y and z negated to reach gsplat's OpenCV basis;
    negating z alone leaves a determinant of -1, i.e. a mirrored image.
    'opencv' passes the basis through unchanged."""


class GaussianSensor(SensorBase):
    """
    Standalone sensor for hybrid 3DGS + MuJoCo rendering.

    Usage:
        config = GaussianSensorConfig(
            width=960,
            height=640,
            background_ply_path="data/pretrained/kitchen/point_cloud/iteration_30000/point_cloud.ply"
        )
        sensor = GaussianSensor(config)

        # In simulation loop:
        rgb = sensor.render(model, data, camera_name="main_camera")

    For mjlab integration, use mugs_mjlab.sensors.GaussianSensorMjlab instead.
    """

    def __init__(self, config: GaussianSensorConfig):
        super().__init__()

        if not GSPLAT_AVAILABLE:
            raise ImportError("gsplat is required for GaussianSensor")

        self.cfg = config
        self.device = torch.device(config.device)

        # Load 3DGS background
        if config.background_ply_path is not None:
            print(f"[GaussianSensor] Loading background: {config.background_ply_path}")
            self.background_gaussians = self._load_official_ply(config.background_ply_path)
            print(f"  ✓ Loaded {len(self.background_gaussians['means']):,} Gaussians")
        else:
            self.background_gaussians = None
            print("[GaussianSensor] No background scene loaded")

        # Background cache
        self._cached_background: Optional[np.ndarray] = None
        self._cached_background_depth: Optional[np.ndarray] = None
        self._cache_camera_params: Optional[Dict] = None

        # RoboSnap: a MuJoCo Renderer owns a GL context, so build it once and
        # reuse it across steps instead of per render() call.
        self._renderer: Optional[mujoco.Renderer] = None
        self._renderer_key = None
        self._scene_option: Optional[mujoco.MjvOption] = None
        self._last_bg_alpha: Optional[np.ndarray] = None

        print(f"[GaussianSensor] Initialized:")
        print(f"  - Resolution: {config.width}×{config.height}")
        print(f"  - Mode: {config.render_mode}")
        print(f"  - Device: {self.device}")

    def render(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera_name: str,
        return_components: bool = False,
        camera_params: Optional[Dict] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Render hybrid RGB image.

        Args:
            model: MuJoCo model
            data: MuJoCo data (current simulation state)
            camera_name: Camera name in MuJoCo model
            return_components: If True, return dict with background/foreground/mask
            camera_params: Optional external camera params (from cameras.json).
                          If provided, uses these for 3DGS rendering instead of MuJoCo camera.
                          Expected keys: position, rotation, fx, fy, width, height

        Returns:
            RGB image (H, W, 3) in range [0, 255] uint8
            or dict with components if return_components=True
        """
        if self.cfg.render_mode == "mujoco_only":
            return self._render_mujoco_only(model, data, camera_name)
        elif self.cfg.render_mode == "3dgs_only":
            return self._render_3dgs_only(model, data, camera_name, camera_params)
        else:  # hybrid
            return self._render_hybrid(model, data, camera_name, return_components, camera_params)

    def _render_hybrid(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera_name: str,
        return_components: bool,
        camera_params: Optional[Dict] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Hybrid rendering: 3DGS background + MuJoCo foreground."""

        # 1. Render 3DGS background (cached if static)
        if camera_params is None:
            camera_params = self._extract_camera_params(model, data, camera_name)
        else:
            # Normalize external camera params to internal format
            camera_params = self._normalize_camera_params(camera_params)

        cached = (self.cfg.cache_background
                  and self._cached_background is not None
                  and self._cameras_equal(camera_params, self._cache_camera_params))
        if cached:
            background_rgb = self._cached_background
            background_depth = self._cached_background_depth
        else:
            background_rgb, background_depth = self._render_3dgs_background(camera_params)
            if self.cfg.cache_background:
                self._cached_background = background_rgb
                self._cached_background_depth = background_depth
                self._cache_camera_params = camera_params

        # 2. Render MuJoCo foreground + segmentation (+ depth when needed)
        foreground_rgb, seg_ids, foreground_depth = self._render_mujoco_foreground(
            model, data, camera_name, want_depth=self.cfg.depth_composite
        )

        # 3. Extract robot mask
        robot_mask = self._create_robot_mask(seg_ids, model)

        # Re3Sim substitution: hand these geoms' pixels to the splat, but only
        # where the splat actually reconstructed something.
        splat_painted = np.zeros_like(robot_mask, dtype=bool)
        if self.cfg.splat_geom_ids:
            for gid in self.cfg.splat_geom_ids:
                splat_painted |= seg_ids == gid
            if self._last_bg_alpha is not None:
                splat_painted &= self._last_bg_alpha > 0.5
            robot_mask = robot_mask & ~splat_painted

        # 4. Composite
        composite_rgb, visible = self._composite_images(
            background_rgb, foreground_rgb, robot_mask,
            background_depth=background_depth, foreground_depth=foreground_depth,
        )

        if return_components:
            return {
                "rgb": composite_rgb,
                "background": background_rgb,
                "foreground": foreground_rgb,
                "mask": robot_mask,
                "splat_painted": splat_painted,
                "seg_ids": seg_ids,
                "visible": visible,
                "background_depth": background_depth,
                "foreground_depth": foreground_depth,
            }
        else:
            return composite_rgb

    def _render_3dgs_only(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera_name: str,
        camera_params: Optional[Dict] = None,
    ) -> np.ndarray:
        """Render only 3DGS background."""
        if camera_params is None:
            camera_params = self._extract_camera_params(model, data, camera_name)
        else:
            camera_params = self._normalize_camera_params(camera_params)
        return self._render_3dgs_background(camera_params)[0]

    def _render_mujoco_only(
        self, model: mujoco.MjModel, data: mujoco.MjData, camera_name: str
    ) -> np.ndarray:
        """Render only MuJoCo (no 3DGS)."""
        foreground_rgb, _, _ = self._render_mujoco_foreground(model, data, camera_name)
        return foreground_rgb

    # ─────────────────────────────────────────────────────────────
    # Helper Methods: 3DGS Rendering
    # ─────────────────────────────────────────────────────────────

    def _load_official_ply(self, ply_path: Path) -> Dict[str, torch.Tensor]:
        """Load official 3DGS PLY format with SH coefficients."""
        plydata = PlyData.read(ply_path)
        vertex = plydata["vertex"]

        # Positions
        positions = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)

        # SH DC → RGB
        SH_C0 = 0.28209479177387814
        sh_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1)
        colors = 0.5 + SH_C0 * sh_dc
        colors = np.clip(colors, 0, 1)

        # Opacity (sigmoid)
        opacities = 1 / (1 + np.exp(-vertex["opacity"]))

        # Scales (exp)
        scales = np.exp(np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1))

        # Quaternions (normalize)
        quats = np.stack(
            [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=1
        )
        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)

        # RoboSnap: keep the higher SH bands when the PLY carries them. gsplat
        # wants (N, (deg+1)^2, 3); f_rest_* is stored channel-major.
        sh = None
        sh_degree = 0
        if self.cfg.use_sh:
            names = {p.name for p in vertex.properties}
            rest = sorted((n for n in names if n.startswith("f_rest_")),
                          key=lambda n: int(n.rsplit("_", 1)[1]))
            if rest:
                r = np.stack([np.asarray(vertex[n]) for n in rest], axis=1)
                r = r.reshape(len(r), 3, r.shape[1] // 3)
                sh = np.concatenate([sh_dc[:, :, None], r], axis=2).transpose(0, 2, 1)
                sh_degree = int(round(sh.shape[1] ** 0.5)) - 1

        # Convert to torch tensors on device
        out = {
            "means": torch.from_numpy(positions).float().to(self.device),
            "colors": torch.from_numpy(colors).float().to(self.device),
            "opacities": torch.from_numpy(opacities).float().to(self.device),
            "scales": torch.from_numpy(scales).float().to(self.device),
            "quats": torch.from_numpy(quats).float().to(self.device),
            "sh": None,
            "sh_degree": sh_degree,
        }
        if sh is not None:
            out["sh"] = torch.from_numpy(np.ascontiguousarray(sh)).float().to(self.device)
        return out

    def _extract_camera_params(
        self, model: mujoco.MjModel, data: mujoco.MjData, camera_name: str
    ) -> Dict:
        """Extract camera parameters from MuJoCo."""
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

        # Position & rotation
        camera_pos = data.cam_xpos[camera_id].copy()
        camera_mat = data.cam_xmat[camera_id].reshape(3, 3).copy()

        # Intrinsics. RoboSnap fix: model.cam_fovy is ALWAYS degrees, whatever
        # <compiler angle="..."/> says, and it is the *vertical* FOV, so both
        # focal lengths derive from the image height (square pixels).
        fov_y_radians = np.deg2rad(float(model.cam_fovy[camera_id]))
        fy = (self.cfg.height / 2) / np.tan(fov_y_radians / 2)
        fx = fy
        cx = self.cfg.width / 2
        cy = self.cfg.height / 2

        return {
            "position": camera_pos,
            "rotation_matrix": camera_mat,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        }

    def _normalize_camera_params(self, external_params: Dict) -> Dict:
        """
        Normalize external camera params (e.g., from cameras.json) to internal format.

        External format (cameras.json):
            position: [x, y, z]
            rotation: [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
            fx, fy: focal lengths
            width, height: original resolution

        Internal format:
            position: np.array
            rotation_matrix: np.array (3x3)
            fx, fy: scaled to cfg.width/cfg.height
            cx, cy: principal point
        """
        # Scale focal lengths to target resolution
        orig_width = external_params.get("width", self.cfg.width)
        orig_height = external_params.get("height", self.cfg.height)

        fx = external_params["fx"] * (self.cfg.width / orig_width)
        fy = external_params["fy"] * (self.cfg.height / orig_height)

        return {
            "position": np.array(external_params["position"], dtype=np.float32),
            "rotation_matrix": np.array(external_params["rotation"], dtype=np.float32),
            "fx": fx,
            "fy": fy,
            "cx": self.cfg.width / 2,
            "cy": self.cfg.height / 2,
        }

    def _render_3dgs_background(self, camera_params: Dict):
        """Render 3DGS background using gsplat.

        RoboSnap: returns (rgb uint8, depth float32) - the depth channel feeds
        depth-aware compositing. Depth is 0 where nothing was splatted.
        """
        if self.background_gaussians is None:
            # No background, return black
            return (np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8),
                    np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32))

        # Build view matrix (world → camera).
        # RoboSnap fix: MuJoCo cameras look down -Z with +Y up; gsplat/OpenCV
        # want +Z forward with +Y down, so y AND z flip. Upstream flipped only
        # z, which makes det(R) = -1 and mirrors the background vertically.
        R_cam = np.asarray(camera_params["rotation_matrix"], dtype=np.float64).copy()
        if self.cfg.camera_convention == "mujoco":
            R_cam = R_cam @ np.diag([1.0, -1.0, -1.0])

        R = R_cam.T  # Transpose for world→camera
        t = -R @ camera_params["position"]
        view_matrix = np.eye(4, dtype=np.float32)
        view_matrix[:3, :3] = R
        view_matrix[:3, 3] = t

        viewmat = torch.from_numpy(view_matrix).to(self.device)

        # Camera intrinsics
        fx, fy = float(camera_params["fx"]), float(camera_params["fy"])
        cx, cy = float(camera_params["cx"]), float(camera_params["cy"])
        K = torch.tensor(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=self.device
        )

        # Render with gsplat. RoboSnap: full SH when the PLY has it, and
        # RGB+ED so the depth channel comes back for occlusion testing.
        sh = self.background_gaussians.get("sh")
        colors = sh if sh is not None else self.background_gaussians["colors"]
        kwargs = {} if sh is None else {"sh_degree": self.background_gaussians["sh_degree"]}
        with torch.no_grad():
            render_out, alphas, _ = rasterization(
                means=self.background_gaussians["means"],
                quats=self.background_gaussians["quats"],
                scales=self.background_gaussians["scales"],
                opacities=self.background_gaussians["opacities"],
                colors=colors,
                viewmats=viewmat[None],
                Ks=K[None],
                width=self.cfg.width,
                height=self.cfg.height,
                render_mode="RGB+ED",
                packed=False,
                **kwargs,
            )

        rgb = render_out[0, :, :, :3].clamp(0, 1)
        depth = render_out[0, :, :, 3]
        self._last_bg_alpha = alphas[0, :, :, 0].detach().cpu().numpy()
        return ((rgb.cpu().numpy() * 255).astype(np.uint8),
                depth.detach().cpu().numpy().astype(np.float32))

    # ─────────────────────────────────────────────────────────────
    # Helper Methods: MuJoCo Rendering
    # ─────────────────────────────────────────────────────────────

    def _get_renderer(self, model: mujoco.MjModel) -> mujoco.Renderer:
        """RoboSnap: one Renderer (one GL context) reused across steps."""
        key = (id(model), self.cfg.height, self.cfg.width)
        if self._renderer is None or self._renderer_key != key:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(model, self.cfg.height, self.cfg.width)
            self._renderer_key = key
        if self._scene_option is None:
            self._scene_option = mujoco.MjvOption()
            mujoco.mjv_defaultOption(self._scene_option)
            if self.cfg.disable_rangefinder:
                self._scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
        return self._renderer

    def _render_mujoco_foreground(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera_name: str,
        want_depth: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Render MuJoCo RGB, segmentation ids, and optionally metric depth."""
        renderer = self._get_renderer(model)

        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera_name, scene_option=self._scene_option)

        # RGB
        rgb = renderer.render()

        # Segmentation
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=camera_name, scene_option=self._scene_option)
        seg = renderer.render()
        renderer.disable_segmentation_rendering()

        seg_ids = seg[:, :, 0].astype(np.int32)

        depth = None
        if want_depth:
            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera=camera_name, scene_option=self._scene_option)
            depth = renderer.render().astype(np.float32).copy()
            renderer.disable_depth_rendering()

        return rgb, seg_ids, depth

    def close(self):
        """Release the GL context. Safe to call more than once."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
            self._renderer_key = None

    def _create_robot_mask(self, seg_ids: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
        """Create binary mask for robot geoms."""
        mask = np.zeros_like(seg_ids, dtype=np.uint8)

        # Use geom IDs if provided (useful when geoms lack names)
        if self.cfg.robot_geom_ids is not None:
            for geom_id in self.cfg.robot_geom_ids:
                if 0 <= geom_id < model.ngeom:
                    mask[seg_ids == geom_id] = 1
        else:
            # Fall back to geom names
            for geom_name in self.cfg.robot_geom_names:
                geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                if geom_id >= 0:
                    mask[seg_ids == geom_id] = 1

        return mask

    def _composite_images(
        self,
        background_rgb: np.ndarray,
        foreground_rgb: np.ndarray,
        mask: np.ndarray,
        background_depth: Optional[np.ndarray] = None,
        foreground_depth: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Blend foreground over background.

        Upstream behaviour is a binary mask: masked pixels always win, so a hand
        that is physically *behind* a splatted table edge still draws in front of
        it. With cfg.depth_composite the mask is intersected with a depth test
        against the splat's own depth, which restores occlusion.

        Returns (composite uint8, visible bool mask).
        """
        visible = mask.astype(bool)
        if (self.cfg.depth_composite
                and background_depth is not None and foreground_depth is not None):
            bg_valid = background_depth > 0.0
            if self._last_bg_alpha is not None:
                bg_valid &= self._last_bg_alpha > 0.02
            in_front = foreground_depth <= background_depth * self.cfg.depth_tolerance
            depth_visible = visible & (~bg_valid | in_front)
            # Guard against a bad alignment hiding the robot completely: if the
            # depth test erases almost everything, fall back to the plain mask.
            if visible.mean() > 0.001 and depth_visible.mean() < 0.15 * visible.mean():
                print("[GaussianSensor] depth test rejected "
                      f"{100 * (1 - depth_visible.mean() / max(visible.mean(), 1e-9)):.0f}% "
                      "of the foreground - falling back to the mask")
            else:
                visible = depth_visible

        # The mask is binary, so select instead of blending in float - at
        # 720x1280 the float path costs more than the splat render itself.
        composite = np.where(visible[:, :, None], foreground_rgb, background_rgb)
        return composite.astype(np.uint8, copy=False), visible

    # ─────────────────────────────────────────────────────────────
    # Properties (for base class compatibility)
    # ─────────────────────────────────────────────────────────────

    @property
    def width(self) -> int:
        """Sensor width in pixels."""
        return self.cfg.width

    @property
    def height(self) -> int:
        """Sensor height in pixels."""
        return self.cfg.height

    def get_observation_space(self) -> Dict:
        """Get observation space specification."""
        return {
            "shape": (self.cfg.height, self.cfg.width, 3),
            "dtype": np.uint8,
            "low": 0,
            "high": 255,
        }

    # ─────────────────────────────────────────────────────────────
    # Utility Methods
    # ─────────────────────────────────────────────────────────────

    def _cameras_equal(self, cam1: Optional[Dict], cam2: Optional[Dict]) -> bool:
        """Check if two camera parameter dicts are equal."""
        if cam1 is None or cam2 is None:
            return False

        return (
            np.allclose(cam1["position"], cam2["position"], atol=1e-6)
            and np.allclose(cam1["rotation_matrix"], cam2["rotation_matrix"], atol=1e-6)
            and abs(cam1["fx"] - cam2["fx"]) < 1e-3
            and abs(cam1["fy"] - cam2["fy"]) < 1e-3
        )

    def clear_cache(self):
        """Clear cached background."""
        self._cached_background = None
        self._cache_camera_params = None
        print("[GaussianSensor] Cache cleared")

    def get_stats(self) -> Dict:
        """Get sensor statistics."""
        stats = {
            "config": {
                "resolution": (self.cfg.width, self.cfg.height),
                "mode": self.cfg.render_mode,
                "device": str(self.device),
            },
            "background_loaded": self.background_gaussians is not None,
            "cache_active": self._cached_background is not None,
        }

        if self.background_gaussians is not None:
            stats["n_gaussians"] = len(self.background_gaussians["means"])

        return stats

    def get_info(self) -> Dict:
        """Get sensor info (alias for get_stats)."""
        return self.get_stats()
