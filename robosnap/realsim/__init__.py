"""Real-to-sim runtime: place a robot in a reconstructed scene, drive it with a
whole-body controller, and render it against the reconstruction's splat.

Everything robot-specific is data, not code - see configs/robots/*.yaml and
`RobotSpec` below.
"""
from .robot_spec import RobotSpec, load_robot

__all__ = ["RobotSpec", "load_robot"]
