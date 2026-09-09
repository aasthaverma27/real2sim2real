"""One robot descriptor, loaded once and shared by every realsim entry point.

The point of this module is that nothing downstream needs to know which robot
it is driving. A descriptor names the MJCF, the joint order the controller's
observation is indexed by, the gains, and where the head cameras sit; the sim,
the WBC loop, the hybrid viewer and the camera publisher all read it. Adding a
robot is a new YAML file, not a new code path.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
ROBOT_DIR = ROOT / "configs" / "robots"


def _resolve(p: str | Path) -> Path:
    """Resolve a descriptor path.

    Expands ${VAR} and ~ first, so a descriptor can name an external tool
    location through configs/paths.env instead of hardcoding one machine's
    layout. Repo-relative paths stay portable; absolute ones are left alone.
    """
    p = Path(os.path.expanduser(os.path.expandvars(str(p))))
    return p if p.is_absolute() else (ROOT / p)


@dataclass
class RobotSpec:
    name: str
    mjcf: Path
    base_joint: str
    root_body: str
    n_robot_joints: int
    body_joints: list[str]
    hand_joints: list[str]
    wbc: dict[str, Any]
    cameras: dict[str, Any]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    # ---- derived ----------------------------------------------------------
    @property
    def n_body(self) -> int:
        return len(self.body_joints)

    @property
    def n_hand(self) -> int:
        return len(self.hand_joints)

    @property
    def n_act(self) -> int:
        """Joints the WBC emits: legs + waist."""
        return int(self.wbc["n_act"])

    @property
    def single_obs(self) -> int:
        """cmd(3) + height(1) + rpy(3) + omega(3) + gravity(3) + q + dq + action."""
        return 13 + 2 * self.n_body + self.n_act

    def default_pose(self) -> np.ndarray:
        """Full body-length default, zero-padded past the WBC's own slice."""
        d = np.asarray(self.wbc["default_pose"], dtype=np.float32)
        if len(d) == self.n_body:
            return d
        if len(d) != self.n_act:
            raise ValueError(
                f"{self.name}: wbc.default_pose has {len(d)} entries, expected "
                f"{self.n_act} (the WBC's slice) or {self.n_body} (all joints)")
        return np.concatenate([d, np.zeros(self.n_body - self.n_act, np.float32)])

    def gains(self) -> tuple[np.ndarray, np.ndarray]:
        kp = np.asarray(self.wbc["kp"], dtype=float)
        kd = np.asarray(self.wbc["kd"], dtype=float)
        for nm, v in (("kp", kp), ("kd", kd)):
            if len(v) != self.n_body:
                raise ValueError(f"{self.name}: wbc.{nm} has {len(v)} entries, "
                                 f"expected {self.n_body}")
        return kp, kd

    def wbc_models(self) -> tuple[Path, Path]:
        return (_resolve(self.wbc["balance_onnx"]), _resolve(self.wbc["walk_onnx"]))

    # ---- model lookups ----------------------------------------------------
    def joint_addr(self, model, prefix: str = "") -> dict[str, np.ndarray]:
        """qpos/qvel/actuator indices for the body and hand joints, in order.

        Accepts both `<prefix><name>_joint` and `<prefix><name>`, because
        add_robot.py namespaces joints while make_wbc_scene.py deliberately
        does not - the reference sim requires the robot's own names.
        """
        import mujoco

        def one(kind, name):
            for cand in (f"{prefix}{name}_joint", f"{prefix}{name}"):
                i = mujoco.mj_name2id(model, kind, cand)
                if i >= 0:
                    return i
            raise SystemExit(f"{self.name}: no {kind.name} '{prefix}{name}' in the model")

        jb = [one(mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.body_joints]
        jh = [one(mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.hand_joints]
        ab = [one(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.body_joints]
        ah = [one(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.hand_joints]
        base = one(mujoco.mjtObj.mjOBJ_JOINT, self.base_joint.removesuffix("_joint"))
        return {
            "body_q": np.array([model.jnt_qposadr[i] for i in jb]),
            "body_v": np.array([model.jnt_dofadr[i] for i in jb]),
            "body_a": np.array(ab),
            "hand_q": np.array([model.jnt_qposadr[i] for i in jh]),
            "hand_v": np.array([model.jnt_dofadr[i] for i in jh]),
            "hand_a": np.array(ah),
            "base_q": int(model.jnt_qposadr[base]),
            "base_v": int(model.jnt_dofadr[base]),
        }


def load_robot(name_or_path: str | Path) -> RobotSpec:
    """Load `configs/robots/<name>.yaml`, or an explicit path to one."""
    p = Path(name_or_path)
    if not p.suffix:
        p = ROBOT_DIR / f"{p.name}.yaml"
    p = _resolve(p)
    if not p.is_file():
        have = sorted(q.stem for q in ROBOT_DIR.glob("*.yaml"))
        raise SystemExit(f"no robot descriptor at {p}. Available: {', '.join(have) or 'none'}")
    d = yaml.safe_load(p.read_text(encoding="utf-8"))

    mjcf = _resolve(d["mjcf"])
    if not mjcf.is_file():
        raise SystemExit(f"{p.name}: mjcf not found at {mjcf}")

    sim = d.get("sim", {})
    hands = [f"{side}_hand_{f}"
             for side in d.get("hand_sides", [])
             for f in d.get("hand_finger_order", [])]
    return RobotSpec(
        name=d.get("name", p.stem),
        mjcf=mjcf,
        base_joint=sim.get("base_joint", "floating_base_joint"),
        root_body=sim.get("root_body", "pelvis"),
        n_robot_joints=int(sim.get("n_robot_joints", 0)),
        body_joints=list(d["body_joints"]),
        hand_joints=hands,
        wbc=d.get("wbc", {}),
        cameras=d.get("cameras", {}),
        raw=d,
    )
