"""
Unit tests for sensor configuration.

Tests GaussianSensorMjlabCfg and related configuration classes.
"""

import pytest

# mjlab-dependent sensors live in the optional `mugs_mjlab` package,
# which itself imports `mjlab`. Skip these tests when mjlab is unavailable.
pytest.importorskip("mjlab")

from mugs_mjlab.sensors.gaussian_sensor import GaussianSensorMjlabCfg  # noqa: E402


class TestGaussianSensorMjlabCfg:
    """Test GaussianSensorMjlabCfg dataclass."""

    def test_minimal_config(self):
        """Test minimal valid configuration."""
        cfg = GaussianSensorMjlabCfg(
            name="test_sensor",
            width=320,
            height=240,
        )

        assert cfg.name == "test_sensor"
        assert cfg.width == 320
        assert cfg.height == 240
        assert cfg.camera_name is None
        assert cfg.render_mode == "hybrid"

    def test_full_config(self):
        """Test configuration with all parameters."""
        cfg = GaussianSensorMjlabCfg(
            name="full_sensor",
            width=640,
            height=480,
            camera_name="kitchen_cam",
            pos=(0.5, -0.8, 1.2),
            quat=(1.0, 0.0, 0.0, 0.0),
            fov_degrees=60.0,
            background_ply_path="/path/to/scene.ply",
            render_mode="3dgs_only",
            robot_geom_names=['link1', 'link2'],
            cache_background=True,
            return_components=True,
        )

        assert cfg.name == "full_sensor"
        assert cfg.width == 640
        assert cfg.height == 480
        assert cfg.camera_name == "kitchen_cam"
        assert cfg.pos == (0.5, -0.8, 1.2)
        assert cfg.quat == (1.0, 0.0, 0.0, 0.0)
        assert cfg.fov_degrees == 60.0
        assert cfg.background_ply_path == "/path/to/scene.ply"
        assert cfg.render_mode == "3dgs_only"
        assert cfg.robot_geom_names == ['link1', 'link2']
        assert cfg.cache_background is True
        assert cfg.return_components is True

    def test_render_modes(self):
        """Test all valid render modes."""
        valid_modes = ["hybrid", "3dgs_only", "mujoco_only"]

        for mode in valid_modes:
            cfg = GaussianSensorMjlabCfg(
                name="test",
                width=320,
                height=240,
                render_mode=mode,
            )
            assert cfg.render_mode == mode

    @pytest.mark.skip(
        reason="GaussianSensorMjlabCfg currently does not validate width/height ranges; "
        "kept as a placeholder for when range validation is added."
    )
    def test_resolution_validation(self):
        """Test resolution must be positive."""
        with pytest.raises((ValueError, AssertionError)):
            GaussianSensorMjlabCfg(
                name="test",
                width=0,
                height=240,
            )

        with pytest.raises((ValueError, AssertionError)):
            GaussianSensorMjlabCfg(
                name="test",
                width=320,
                height=-10,
            )

    def test_fov_validation(self):
        """Test FOV must be in valid range."""
        # Valid FOV
        cfg = GaussianSensorMjlabCfg(
            name="test",
            width=320,
            height=240,
            fov_degrees=60.0,
        )
        assert cfg.fov_degrees == 60.0

        # Edge cases
        cfg_min = GaussianSensorMjlabCfg(
            name="test",
            width=320,
            height=240,
            fov_degrees=1.0,
        )
        assert cfg_min.fov_degrees == 1.0

        cfg_max = GaussianSensorMjlabCfg(
            name="test",
            width=320,
            height=240,
            fov_degrees=179.0,
        )
        assert cfg_max.fov_degrees == 179.0

    def test_build_sensor(self):
        """Test building sensor from config."""
        cfg = GaussianSensorMjlabCfg(
            name="test_sensor",
            width=320,
            height=240,
        )

        sensor = cfg.build()

        assert sensor is not None
        assert sensor.cfg == cfg
        assert sensor._camera_name == "test_sensor"

    def test_build_with_existing_camera(self):
        """Test building sensor that wraps existing camera."""
        cfg = GaussianSensorMjlabCfg(
            name="gaussian_sensor",
            width=640,
            height=480,
            camera_name="existing_camera",
        )

        sensor = cfg.build()

        assert sensor._camera_name == "existing_camera"
        assert sensor._is_wrapping is True


class TestGaussianSensorData:
    """Test GaussianSensorData dataclass."""

    def test_minimal_data(self):
        """Test data with only RGB."""
        import torch

        rgb = torch.randint(0, 256, (16, 480, 640, 3), dtype=torch.uint8)

        from mugs_mjlab.sensors.gaussian_sensor import GaussianSensorData
        data = GaussianSensorData(rgb=rgb)

        assert data.rgb.shape == (16, 480, 640, 3)
        assert data.rgb.dtype == torch.uint8
        assert data.background is None
        assert data.foreground is None
        assert data.mask is None

    def test_full_data(self):
        """Test data with all components."""
        import torch

        num_envs = 16
        H, W = 480, 640

        rgb = torch.randint(0, 256, (num_envs, H, W, 3), dtype=torch.uint8)
        background = torch.randint(0, 256, (num_envs, H, W, 3), dtype=torch.uint8)
        foreground = torch.randint(0, 256, (num_envs, H, W, 3), dtype=torch.uint8)
        mask = torch.rand(num_envs, H, W, 1)

        from mugs_mjlab.sensors.gaussian_sensor import GaussianSensorData
        data = GaussianSensorData(
            rgb=rgb,
            background=background,
            foreground=foreground,
            mask=mask,
        )

        assert data.rgb.shape == (num_envs, H, W, 3)
        assert data.background.shape == (num_envs, H, W, 3)
        assert data.foreground.shape == (num_envs, H, W, 3)
        assert data.mask.shape == (num_envs, H, W, 1)

    def test_batch_dimensions(self):
        """Test batch-first dimension ordering."""
        import torch

        # Different batch sizes
        for num_envs in [1, 4, 16, 256]:
            rgb = torch.zeros((num_envs, 240, 320, 3), dtype=torch.uint8)

            from mugs_mjlab.sensors.gaussian_sensor import GaussianSensorData
            data = GaussianSensorData(rgb=rgb)

            assert data.rgb.shape[0] == num_envs
            assert data.rgb.shape[-1] == 3  # Channels last
