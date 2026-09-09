#!/usr/bin/env python3
"""Play a trained GR00T checkpoint in the photoreal scene, with the WBC on legs.

This is the AndroidTwin-style harness for our scenes: policy and whole-body
controller run in the *same process* as MuJoCo, and every control step is
rendered through MuGS's GaussianSensor - which is the only arrangement MuGS
supports, since the sensor draws whatever mj_data the process already owns.

Why the WBC is not optional: the checkpoint's action space is

    left_arm 7, right_arm 7, left_hand 7, right_hand 7, waist 3,
    base_height_command 1, navigate_command 3

with no leg actions at all. It steers the base with a velocity command; the
Balance/Walk networks turn that into leg targets. So the loop is

    ego_view -> GR00T server -> arms/hands + nav cmd -> WBC -> legs -> torques

Nothing here retrains or edits either policy: the GR00T checkpoint is served by
the same run_gr00t_server as always, and the WBC nets are their shipped ONNX.

    bash scripts/run_play_groot_wbc.sh --scene-dir outputs/video_scene
"""
from __future__ import annotations

import argparse
import os
import collections
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party" / "MuGS" / "src"))

from mugs_render import (apply_vram_cap, foreground_geom_ids, inject_capture_camera,
                         newest_background, real_intrinsics)

# Canonical Unitree 29-DOF order - the order the WBC observation is indexed by.
BODY = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
N_BODY, N_ACT = 29, 15                      # 15 = 12 legs + 3 waist, what the WBC emits
# GR00T's per-hand vector follows the robot model's hand group order (index,
# middle, thumb), which is NOT the MJCF's joint order (thumb first).
GROOT_FINGERS = ["index_0", "index_1", "middle_0", "middle_1",
                 "thumb_0", "thumb_1", "thumb_2"]

DEFAULT_15 = np.array([-0.1, 0, 0, 0.3, -0.2, 0,
                       -0.1, 0, 0, 0.3, -0.2, 0,
                       0, 0, 0], dtype=np.float32)
DEFAULT_29 = np.concatenate([DEFAULT_15, np.zeros(N_BODY - N_ACT, np.float32)])
KP = np.array([150, 150, 150, 200, 40, 40, 150, 150, 150, 200, 40, 40,
               250, 250, 250, 100, 100, 40, 40, 20, 20, 20,
               100, 100, 40, 40, 20, 20, 20], float)
KD = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5,
               5, 5, 2, 2, 2, 2, 2, 5, 5, 2, 2, 2, 2, 2], float)
HAND_KP, HAND_KD = 2.0, 0.1

ANG_VEL_SCALE, DOF_POS_SCALE, DOF_VEL_SCALE = 0.5, 1.0, 0.05
ACTION_SCALE = 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.5], np.float32)
OBS_HIST = 6
SINGLE_OBS = 13 + 2 * N_BODY + N_ACT        # 86
LANG = ("observe the entire scene first. identify the red cube, blue cube, and green "
        "cube. then pick up each cube and place it into the basket with the matching "
        "color. place the red cube in the red basket, the blue cube in the blue "
        "basket, and the green cube in the green basket. ensure every cube is placed "
        "in the correct colored basket.")


def gravity_orientation(q):
    w, x, y, z = q
    return np.array([2 * (-x * z + w * y), -2 * (y * z + w * x),
                     -(w * w - x * x - y * y + z * z)], np.float32)


class Wbc:
    """The shipped GEAR WBC nets, driven from MuJoCo state instead of DDS."""

    def __init__(self, balance: Path, walk: Path):
        import onnxruntime as ort
        o = ort.SessionOptions()
        o.intra_op_num_threads = 2
        self.balance = ort.InferenceSession(str(balance), o, providers=["CPUExecutionProvider"])
        self.walk = ort.InferenceSession(str(walk), o, providers=["CPUExecutionProvider"])
        self.name = self.balance.get_inputs()[0].name
        self.n_obs = self.balance.get_inputs()[0].shape[-1]
        if self.n_obs != SINGLE_OBS * OBS_HIST:
            raise SystemExit(f"WBC expects {self.n_obs} obs, we build {SINGLE_OBS * OBS_HIST}")
        self.hist = collections.deque(maxlen=OBS_HIST)
        self.action = np.zeros(N_ACT, np.float32)

    def step(self, qj, dqj, quat, omega, cmd, height, rpy) -> np.ndarray:
        obs = np.zeros(SINGLE_OBS, np.float32)
        obs[0:3] = cmd * CMD_SCALE
        obs[3] = height
        obs[4:7] = rpy
        obs[7:10] = omega * ANG_VEL_SCALE
        obs[10:13] = gravity_orientation(quat)
        obs[13:13 + N_BODY] = (qj - DEFAULT_29) * DOF_POS_SCALE
        obs[13 + N_BODY:13 + 2 * N_BODY] = dqj * DOF_VEL_SCALE
        obs[13 + 2 * N_BODY:] = self.action
        self.hist.append(obs)
        while len(self.hist) < OBS_HIST:
            self.hist.appendleft(np.zeros_like(obs))
        stack = np.concatenate(self.hist)[None, :]
        # Standing still is a different network from walking - the same split the
        # deployed controller makes.
        net = self.balance if np.linalg.norm(cmd) < 0.05 else self.walk
        self.action = net.run(None, {self.name: stack})[0].squeeze()
        return self.action * ACTION_SCALE + DEFAULT_15


class Groot:
    """The checkpoint, over the same server the deployed client talks to."""

    def __init__(self, host, port, lang, timeout_ms=20000):
        from gr00t.policy.server_client import PolicyClient
        self.client = PolicyClient(host, port, timeout_ms=timeout_ms)
        if not self.client.ping():
            raise SystemExit(f"no GR00T server on {host}:{port}")
        self.lang = lang
        self.chunk, self.idx, self.horizon = None, 0, 0
        print(f"[groot] connected to {host}:{port}")

    @staticmethod
    def _batch(v):
        """(B=1, T=1) in front of every leaf, as the deployed adapter does."""
        return v[None, None, ...] if isinstance(v, np.ndarray) else [[v]]

    def query(self, ego_rgb, state: dict):
        obs = {"video": {"ego_view": self._batch(ego_rgb)},
               "state": {k: self._batch(v.astype(np.float32)) for k, v in state.items()},
               "language": {"annotation.human.task_description": self._batch(self.lang)}}
        self.chunk, _ = self.client.get_action(obs)
        first = next(iter(self.chunk.values()))
        self.horizon = first.shape[1]
        self.idx = 0

    def current(self) -> dict:
        i = min(self.idx, self.horizon - 1)
        return {k: np.asarray(v)[0, i] for k, v in self.chunk.items()}

    def exhausted(self) -> bool:
        return self.chunk is None or self.idx >= self.horizon


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-dir", type=Path, default=ROOT / "outputs" / "video_scene")
    p.add_argument("--mjcf", type=Path, help="default: <scene>/mujoco/scene_with_robot.xml")
    p.add_argument("--ply", type=Path)
    p.add_argument("--wbc-models", type=Path, default=Path(os.environ.get("RS_WBC_MODELS", "/mnt/ssd2/simple_wbc_code/models")))
    p.add_argument("--policy-host", default="localhost")
    p.add_argument("--policy-port", type=int, default=5560)
    p.add_argument("--lang", default=LANG)
    p.add_argument("--prefix", default="g1_")
    p.add_argument("--ego-camera", default="g1_ego_view")
    p.add_argument("--camera", help="render through a camera the MJCF already "
                   "defines, instead of injecting one from gravity_alignment.json. "
                   "outputs/g1_ok's 'cap' was hand-fitted to the photograph; the "
                   "injected one predates that fit and looks from the wrong place.")
    p.add_argument("--fovy", type=float, help="override the vertical FOV used for "
                   "the splat, in degrees. Needed with --camera, because the "
                   "capture calibration in depth/camera.json describes the phone, "
                   "not a camera that was afterwards re-fitted by hand.")
    p.add_argument("--no-policy", action="store_true",
                   help="run the WBC alone: arms and hands stay in the default "
                   "stance and the navigation command is zero, so the balance net "
                   "holds the robot up. No GR00T server needed.")
    p.add_argument("--width", type=int, default=440)
    p.add_argument("--height", type=int, default=780)
    p.add_argument("--ego-width", type=int, default=640)
    p.add_argument("--ego-height", type=int, default=480)
    p.add_argument("--policy-hz", type=float, default=20.0, help="dataset rate")
    p.add_argument("--mode", default="hybrid", choices=["hybrid", "3dgs_only", "mujoco_only"])
    p.add_argument("--record", type=Path, metavar="MP4")
    p.add_argument("--seconds", type=float, default=0.0, help="stop after N s (0 = until ESC)")
    p.add_argument("--no-window", action="store_true", help="record only, no display")
    p.add_argument("--vram-cap-gb", type=float, default=20.0)
    p.add_argument("--viewer", action="store_true",
                   help="also open MuJoCo's interactive viewer on the SAME "
                   "mj_data the WBC is driving, so the orbitable sim and the "
                   "photoreal composite show one simulation instead of two. "
                   "Needs MUJOCO_GL=glfw - the passive viewer wants a real "
                   "window, and EGL cannot give it one.")
    a = p.parse_args()

    import cv2
    import mujoco
    from mugs.sensors.gaussian_sensor import GaussianSensor, GaussianSensorConfig

    mjcf = a.mjcf or a.scene_dir / "mujoco" / "scene_with_robot.xml"
    ply = a.ply or newest_background(a.scene_dir)
    apply_vram_cap(a.vram_cap_gb)

    intr = real_intrinsics(a.scene_dir, a.width, a.height)
    fovy = a.fovy if a.fovy else (intr[2] if intr else 58.0)
    if a.camera:
        cam_name = a.camera
        if a.fovy:
            # real_intrinsics' fx/fy belong to the phone. Recompute them from the
            # FOV actually being rendered, or the splat is drawn at one focal
            # length and the meshes at another and nothing lines up.
            f = (a.height / 2.0) / np.tan(np.deg2rad(a.fovy) / 2.0)
            intr = (f, f, a.fovy)
    else:
        mjcf, cam_name = inject_capture_camera(
            mjcf, a.scene_dir / "gravity_alignment.json", fovy)

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)

    def jadr(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{a.prefix}{name}_joint")
        if jid < 0:
            raise SystemExit(f"no joint '{a.prefix}{name}_joint' in {mjcf.name}")
        return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])

    def aid(name):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{a.prefix}{name}")
        if i < 0:
            raise SystemExit(f"no actuator '{a.prefix}{name}'")
        return i

    body_q = np.array([jadr(n)[0] for n in BODY])
    body_v = np.array([jadr(n)[1] for n in BODY])
    body_a = np.array([aid(n) for n in BODY])
    hands = [f"{s}_hand_{f}" for s in ("left", "right") for f in GROOT_FINGERS]
    hand_q = np.array([jadr(n)[0] for n in hands])
    hand_v = np.array([jadr(n)[1] for n in hands])
    hand_a = np.array([aid(n) for n in hands])
    base_q, base_v = jadr("floating_base")

    # Stand the robot in the WBC's default stance, feet on the floor.
    mujoco.mj_resetData(model, data)
    data.qpos[body_q] = DEFAULT_29
    data.qpos[hand_q] = 0.0
    mujoco.mj_forward(model, data)
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    floor_z = float(model.geom_pos[gid][2]) if gid >= 0 else 0.0
    robot_geoms = [g for g in range(model.ngeom)
                   if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                         int(model.geom_bodyid[g])) or "").startswith(a.prefix)]
    low = float("inf")
    for g in robot_geoms:
        pos, mat = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3)
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            m0 = int(model.geom_dataid[g])
            v0, n0 = int(model.mesh_vertadr[m0]), int(model.mesh_vertnum[m0])
            low = min(low, float((model.mesh_vert[v0:v0 + n0].reshape(-1, 3) @ mat.T + pos)[:, 2].min()))
        else:
            low = min(low, float(pos[2] - model.geom_size[g].max()))
    data.qpos[base_q + 2] += floor_z + 0.002 - low
    mujoco.mj_forward(model, data)
    print(f"[play] stance set, pelvis z={data.qpos[base_q + 2]:+.3f}")

    wbc = Wbc(a.wbc_models / "GR00T-WholeBodyControl-Balance.onnx",
              a.wbc_models / "GR00T-WholeBodyControl-Walk.onnx")
    groot = None if a.no_policy else Groot(a.policy_host, a.policy_port, a.lang)

    fg = foreground_geom_ids(model, {"floor"})
    sensor = GaussianSensor(GaussianSensorConfig(
        width=a.width, height=a.height, background_ply_path=ply, render_mode=a.mode,
        robot_geom_ids=fg, use_sh=True, depth_composite=True, camera_convention="mujoco"))
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        raise SystemExit(f"no camera '{cam_name}' in {mjcf.name}")
    ego = ego_id = None
    if groot is not None:
        ego = mujoco.Renderer(model, a.ego_height, a.ego_width)
        ego_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, a.ego_camera)
        if ego_id < 0:
            raise SystemExit(f"no camera '{a.ego_camera}' - the policy needs its "
                             f"ego view")

    passive = None
    if a.viewer:
        import mujoco.viewer
        passive = mujoco.viewer.launch_passive(model, data,
                                               show_left_ui=False, show_right_ui=False)
        print("[play] MuJoCo viewer attached to the live data")

    writer = None
    if a.record:
        writer = cv2.VideoWriter(str(a.record), cv2.VideoWriter_fourcc(*"mp4v"),
                                 int(a.policy_hz), (a.width, a.height))
    if not a.no_window:
        cv2.namedWindow("GR00T + WBC", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("GR00T + WBC", a.width, a.height)

    ctrl_hz = 50.0
    decim = max(1, round((1 / ctrl_hz) / model.opt.timestep))
    per_action = max(1, round(ctrl_hz / a.policy_hz))     # control steps per action step
    target29 = DEFAULT_29.astype(float).copy()
    hand_target = np.zeros(14)
    nav, height = np.zeros(3, np.float32), 0.74
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    print(f"[play] control {ctrl_hz:.0f} Hz, policy {a.policy_hz:.0f} Hz, "
          f"{per_action} control steps per action")

    def state_dict():
        q = data.qpos[body_q]
        h = data.qpos[hand_q]
        return {"left_leg": q[0:6], "right_leg": q[6:12], "waist": q[12:15],
                "left_arm": q[15:22], "right_arm": q[22:29],
                "left_hand": h[0:7], "right_hand": h[7:14]}

    t_start, step, queries = time.perf_counter(), 0, 0
    try:
        while True:
            if groot is not None and step % per_action == 0:
                if groot.exhausted():
                    ego.update_scene(data, camera=ego_id)
                    groot.query(ego.render().copy(), state_dict())
                    queries += 1
                act = groot.current()
                groot.idx += 1
                target29[15:22] = act["left_arm"]
                target29[22:29] = act["right_arm"]
                hand_target[0:7] = act["left_hand"]
                hand_target[7:14] = act["right_hand"]
                nav = np.asarray(act["navigate_command"], np.float32).reshape(3)
                height = float(np.asarray(act["base_height_command"]).reshape(-1)[0])

            # WBC: legs + waist from the navigation command the checkpoint emitted
            target29[:N_ACT] = wbc.step(
                data.qpos[body_q].astype(np.float32), data.qvel[body_v].astype(np.float32),
                data.qpos[base_q + 3:base_q + 7].astype(np.float32),
                data.qvel[base_v + 3:base_v + 6].astype(np.float32),
                nav, height, np.zeros(3, np.float32))

            for _ in range(decim):
                tau = KP * (target29 - data.qpos[body_q]) - KD * data.qvel[body_v]
                data.ctrl[body_a] = np.clip(tau, lo[body_a], hi[body_a])
                htau = HAND_KP * (hand_target - data.qpos[hand_q]) - HAND_KD * data.qvel[hand_v]
                data.ctrl[hand_a] = np.clip(htau, lo[hand_a], hi[hand_a])
                mujoco.mj_step(model, data)
            step += 1

            if passive is not None:
                if not passive.is_running():
                    break
                passive.sync()

            if step % per_action == 0:
                params = None
                if intr:
                    params = {"position": data.cam_xpos[cam_id].copy(),
                              "rotation": data.cam_xmat[cam_id].reshape(3, 3).copy(),
                              "fx": intr[0], "fy": intr[1],
                              "width": a.width, "height": a.height}
                rgb = sensor.render(model, data, cam_name, camera_params=params)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                tag = "WBC only" if groot is None else "GR00T+WBC"
                cv2.putText(bgr, f"{tag}  nav {nav[0]:+.2f} {nav[1]:+.2f} {nav[2]:+.2f}"
                                 f"  h {height:.2f}  pelvis {data.qpos[base_q+2]:+.3f}"
                                 f"  q{queries}",
                            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                if writer is not None:
                    writer.write(bgr)
                if not a.no_window:
                    cv2.imshow("GR00T + WBC", bgr)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        break
            if a.seconds and time.perf_counter() - t_start > a.seconds:
                break
            if not np.isfinite(data.qpos).all():
                print("[play] simulation diverged")
                break
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
            print(f"[play] wrote {a.record}")
        cv2.destroyAllWindows()
        if passive is not None:
            passive.close()
        sensor.close()
    print(f"[play] {step} control steps, {queries} policy queries, "
          f"pelvis z={data.qpos[base_q + 2]:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
