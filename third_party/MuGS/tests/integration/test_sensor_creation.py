"""
Integration tests for sensor creation and initialization.

Tests the full sensor creation pipeline without actual rendering.
"""

import pytest

# Sensor creation goes through `mugs_mjlab.sensors.gaussian_sensor`,
# which imports `mjlab`. Skip the whole module when mjlab is unavailable.
pytest.importorskip("mjlab")

import mujoco  # noqa: E402


class TestSensorCreation:
    """Test sensor creation and scene integration."""

    @pytest.fixture
    def simple_scene_xml(self):
        """Create a simple MuJoCo scene."""
        return """
        <mujoco model="test">
          <compiler angle="radian"/>
          <worldbody>
            <light pos="0 0 3" dir="0 0 -1"/>
            <geom name="floor" type="plane" size="5 5 0.1"/>
            <body name="box" pos="0 0 0.5">
              <geom name="box_geom" type="box" size="0.1 0.1 0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """

    def test_sensor_edit_spec(self, simple_scene_xml):
        """Test sensor can edit MuJoCo spec."""
        from mugs_mjlab.sensors.gaussian_sensor import GaussianSensorMjlabCfg

        # Create sensor config
        cfg = GaussianSensorMjlabCfg(
            name="test_camera",
            width=320,
            height=240,
            pos=(1.0, -1.0, 1.5),
            quat=(1.0, 0.0, 0.0, 0.0),
            fov_degrees=60.0,
        )

        sensor = cfg.build()

        # Create spec
        spec = mujoco.MjSpec.from_string(simple_scene_xml)

        # Sensor edits spec
        sensor.edit_spec(spec, {})

        # Verify camera was added
        camera = spec.camera("test_camera")
        assert camera is not None
        assert camera.name == "test_camera"

    def test_sensor_wraps_existing_camera(self, simple_scene_xml):
        """Test sensor can wrap existing camera."""
        from mugs_mjlab.sensors.gaussian_sensor import GaussianSensorMjlabCfg

        # Scene with existing camera
        scene_with_cam = """
        <mujoco model="test">
          <worldbody>
            <camera name="existing_cam" pos="1 0 1" xyaxes="0 -1 0 0 0 1"/>
            <geom name="floor" type="plane" size="5 5 0.1"/>
          </worldbody>
        </mujoco>
        """

        # Create sensor that wraps existing camera
        cfg = GaussianSensorMjlabCfg(
            name="wrapper_sensor",
            width=640,
            height=480,
            camera_name="existing_cam",
        )

        sensor = cfg.build()
        assert sensor._is_wrapping is True
        assert sensor._camera_name == "existing_cam"

        # Create spec
        spec = mujoco.MjSpec.from_string(scene_with_cam)

        # Sensor should not add new camera
        initial_ncam = len([c for c in spec.cameras])

        sensor.edit_spec(spec, {})

        final_ncam = len([c for c in spec.cameras])
        assert final_ncam == initial_ncam  # No new camera added

    def test_sensor_compile_model(self, simple_scene_xml):
        """Test full scene compilation with sensor."""
        from mugs_mjlab.sensors.gaussian_sensor import GaussianSensorMjlabCfg

        cfg = GaussianSensorMjlabCfg(
            name="test_camera",
            width=320,
            height=240,
            pos=(0.8, -0.6, 1.0),
            fov_degrees=60.0,
        )

        sensor = cfg.build()

        # Create and edit spec
        spec = mujoco.MjSpec.from_string(simple_scene_xml)
        sensor.edit_spec(spec, {})

        # Compile model
        model = spec.compile()

        assert model is not None
        assert model.ncam > 0

        # Find camera
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "test_camera")
        assert cam_id >= 0

    def test_multiple_sensors(self, simple_scene_xml):
        """Test multiple sensors in same scene."""
        from mugs_mjlab.sensors.gaussian_sensor import GaussianSensorMjlabCfg

        # Create two sensors
        cfg1 = GaussianSensorMjlabCfg(
            name="camera_1",
            width=320,
            height=240,
            pos=(1.0, 0.0, 1.0),
        )

        cfg2 = GaussianSensorMjlabCfg(
            name="camera_2",
            width=640,
            height=480,
            pos=(-1.0, 0.0, 1.0),
        )

        sensor1 = cfg1.build()
        sensor2 = cfg2.build()

        # Create spec
        spec = mujoco.MjSpec.from_string(simple_scene_xml)

        # Both sensors edit spec
        sensor1.edit_spec(spec, {})
        sensor2.edit_spec(spec, {})

        # Compile
        model = spec.compile()

        # Verify both cameras exist
        cam1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "camera_1")
        cam2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "camera_2")

        assert cam1_id >= 0
        assert cam2_id >= 0
        assert cam1_id != cam2_id


class TestSensorRobotGeoms:
    """Test robot geom handling."""

    def test_robot_geom_ids_extraction(self):
        """Test extracting robot geom IDs."""
        from mugs_mjlab.sensors.gaussian_sensor import GaussianSensorMjlabCfg

        scene = """
        <mujoco model="robot">
          <worldbody>
            <geom name="floor" type="plane" size="5 5 0.1"/>
            <body name="robot">
              <geom name="base_link" type="box" size="0.1 0.1 0.1"/>
              <body name="arm">
                <geom name="arm_link" type="capsule" size="0.05" fromto="0 0 0 0.3 0 0"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """

        cfg = GaussianSensorMjlabCfg(
            name="test_camera",
            width=320,
            height=240,
            robot_geom_names=['base_link', 'arm_link'],
        )

        sensor = cfg.build()

        # Create model
        spec = mujoco.MjSpec.from_string(scene)
        sensor.edit_spec(spec, {})
        model = spec.compile()

        # Initialize sensor (mock mjwarp objects)
        class MockMjwarpModel:
            pass

        class MockMjwarpData:
            def __init__(self):
                import torch
                self.qpos = torch.zeros((1, 0))  # 1 env, 0 DOF
                self.cam_xpos = torch.zeros((1, 1, 3))
                self.cam_xmat = torch.eye(3).reshape(9).unsqueeze(0).unsqueeze(0)

        mock_model = MockMjwarpModel()
        mock_data = MockMjwarpData()

        sensor.initialize(model, mock_model, mock_data, "cpu")

        # Verify geom IDs were extracted
        assert len(sensor._robot_geom_ids) == 2

        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "base_link")
        arm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "arm_link")

        assert base_id in sensor._robot_geom_ids
        assert arm_id in sensor._robot_geom_ids
