#!/usr/bin/env python3
"""Run a whole-body controller on any robot, in any reconstructed scene.

The controller, the scene and the photoreal background are three independent
choices here:

    --robot   configs/robots/*.yaml   which robot, its joint order and gains
    --mjcf    any MuJoCo scene        where it stands
    --ply     any 3DGS background     what the world looks like

Nothing about the shipped controller is modified: the two ONNX nets named in
the descriptor run as-is, driven from MuJoCo state instead of DDS. That is the
only arrangement MuGS supports - `sensor.render(model, data, camera)` has no
step function and no notion of another simulator, so it draws whatever mj_data
its own process already owns.

Why the controller is not optional: a humanoid is not held up by a PD hold on
its rest pose. Measured on the G1, the pelvis still drops 0.792 -> 0.189 m in
two seconds with the actuators holding, identical to leaving ctrl at zero. The
balance net is what keeps it standing.

With --policy-host, a GR00T checkpoint served by the stock run_gr00t_server
drives the arms and hands while the controller keeps the legs under it; without
it the arms hold the descriptor's default stance and the balance net runs alone.

    scripts/realsim/run_wbc.py --robot g1_29dof_with_hand \
        --mjcf outputs/myroom/mujoco/scene_g1.xml \
        --ply outputs/myroom/gs_bg_notripod.ply --camera cap --fovy 55.21
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party" / "MuGS" / "src"))

from robosnap.realsim import load_robot


def gravity_orientation(q):
    w, x, y, z = q
    return np.array([2 * (-x * z + w * y), -2 * (y * z + w * x),
                     -(w * w - x * x - y * y + z * z)], np.float32)


class Wbc:
    """The descriptor's ONNX nets, driven from MuJoCo state instead of DDS."""

    def __init__(self, spec):
        import onnxruntime as ort

        balance, walk = spec.wbc_models()
        for p in (balance, walk):
            if not p.is_file():
                raise SystemExit(f"{spec.name}: missing WBC model {p}")
        o = ort.SessionOptions()
        o.intra_op_num_threads = 2
        self.balance = ort.InferenceSession(str(balance), o,
                                            providers=["CPUExecutionProvider"])
        self.walk = ort.InferenceSession(str(walk), o,
                                         providers=["CPUExecutionProvider"])
        self.spec = spec
        self.name = self.balance.get_inputs()[0].name
        n_obs = self.balance.get_inputs()[0].shape[-1]
        want = spec.single_obs * int(spec.wbc["obs_history"])
        if n_obs != want:
            raise SystemExit(
                f"{spec.name}: the balance net wants {n_obs} observations but the "
                f"descriptor builds {want} "
                f"({spec.single_obs} x {spec.wbc['obs_history']}). Check "
                f"body_joints, wbc.n_act and wbc.obs_history.")
        self.hist = collections.deque(maxlen=int(spec.wbc["obs_history"]))
        self.action = np.zeros(spec.n_act, np.float32)
        self.default = spec.default_pose()
        self.cmd_scale = np.asarray(spec.wbc["cmd_scale"], np.float32)

    def step(self, qj, dqj, quat, omega, cmd, height, rpy) -> np.ndarray:
        s, n = self.spec, self.spec.n_body
        obs = np.zeros(s.single_obs, np.float32)
        obs[0:3] = cmd * self.cmd_scale
        obs[3] = height
        obs[4:7] = rpy
        obs[7:10] = omega * float(s.wbc["ang_vel_scale"])
        obs[10:13] = gravity_orientation(quat)
        obs[13:13 + n] = (qj - self.default) * float(s.wbc["dof_pos_scale"])
        obs[13 + n:13 + 2 * n] = dqj * float(s.wbc["dof_vel_scale"])
        obs[13 + 2 * n:] = self.action
        self.hist.append(obs)
        while len(self.hist) < self.hist.maxlen:
            self.hist.appendleft(np.zeros_like(obs))
        stack = np.concatenate(self.hist)[None, :]
        # Standing still is a different network from walking - the same split
        # the deployed controller makes.
        net = self.balance if np.linalg.norm(cmd) < 0.05 else self.walk
        self.action = net.run(None, {self.name: stack})[0].squeeze()
        return self.action * float(s.wbc["action_scale"]) + self.default[:s.n_act]


class Groot:
    """A checkpoint, over the same server the deployed client talks to."""

    def __init__(self, host, port, lang, timeout_ms=20000):
        from gr00t.policy.server_client import PolicyClient
        self.client = PolicyClient(host, port, timeout_ms=timeout_ms)
        if not self.client.ping():
            raise SystemExit(f"no GR00T server on {host}:{port}")
        self.lang = lang
        self.chunk, self.idx, self.horizon = None, 0, 0
        print(f"[wbc] policy connected on {host}:{port}")

    @staticmethod
    def _batch(v):
        return v[None, None, ...] if isinstance(v, np.ndarray) else [[v]]

    def query(self, ego_rgb, state: dict):
        obs = {"video": {"ego_view": self._batch(ego_rgb)},
               "state": {k: self._batch(v.astype(np.float32)) for k, v in state.items()},
               "language": {"annotation.human.task_description": self._batch(self.lang)}}
        self.chunk, _ = self.client.get_action(obs)
        self.horizon = next(iter(self.chunk.values())).shape[1]
        self.idx = 0

    def current(self) -> dict:
        i = min(self.idx, self.horizon - 1)
        return {k: np.asarray(v)[0, i] for k, v in self.chunk.items()}

    def exhausted(self) -> bool:
        return self.chunk is None or self.idx >= self.horizon


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default="g1_29dof_with_hand",
                   help="descriptor in configs/robots/, or a path to one")
    p.add_argument("--mjcf", type=Path, required=True)
    p.add_argument("--ply", type=Path, default=None,
                   help="3DGS background. Omit for --mode mujoco_only.")
    p.add_argument("--scene-dir", type=Path, default=None)
    p.add_argument("--prefix", default="", help="robot namespace in --mjcf "
                   "(add_robot.py uses g1_; build_wbc_scene.py uses none)")
    p.add_argument("--camera", default=None,
                   help="camera to render through. Default: the first one.")
    p.add_argument("--fovy", type=float, default=None)
    p.add_argument("--ego-camera", default=None,
                   help="the policy's view. Default: the descriptor's pinhole.")
    p.add_argument("--policy-host", default=None,
                   help="GR00T server host. Omit to run the controller alone.")
    p.add_argument("--policy-port", type=int, default=5560)
    p.add_argument("--policy-hz", type=float, default=20.0)
    p.add_argument("--lang", default="")
    p.add_argument("--width", type=int, default=540)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--ego-width", type=int, default=640)
    p.add_argument("--ego-height", type=int, default=480)
    p.add_argument("--mode", default="hybrid",
                   choices=["hybrid", "3dgs_only", "mujoco_only"])
    p.add_argument("--viewer", action="store_true",
                   help="also open MuJoCo's interactive viewer on the SAME "
                        "mj_data, so the orbitable sim and the composite show "
                        "one simulation. Needs MUJOCO_GL=glfw.")
    p.add_argument("--record", type=Path, metavar="MP4")
    p.add_argument("--seconds", type=float, default=0.0, help="wall-clock, 0 = until ESC")
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--publish-state", type=int, nargs="?", const=5601,
                   metavar="PORT",
                   help="broadcast this process's robot pose as UDP datagrams in "
                        "the same format scripts/wbc_state_relay.py uses, so a "
                        "camera publisher or hybrid viewer started with "
                        "--mirror PORT --mirror-frame identity shows THIS "
                        "simulation instead of stepping its own.")
    p.add_argument("--vram-cap-gb", type=float, default=22.0)
    p.add_argument("--no-randomize", dest="randomize", action="store_false",
                   help="keep the fixed table layout; by default every reset "
                        "(backspace in the viewer) reshuffles the objects.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--fruits", default="",
                   help="plate+basket scenes: comma-separated fruits to put on "
                        "the plate (e.g. mango,dragon_fruit); the rest are "
                        "parked off the table. Default: all.")
    p.add_argument("--reach-min", type=float, default=0.36,
                   help="closest an object may spawn to the robot base, metres")
    p.add_argument("--reach-max", type=float, default=0.52)
    a = p.parse_args()

    import cv2
    import mujoco
    from mugs_render import apply_vram_cap, foreground_geom_ids

    spec = load_robot(a.robot)
    apply_vram_cap(a.vram_cap_gb)

    model = mujoco.MjModel.from_xml_path(str(a.mjcf))
    data = mujoco.MjData(model)
    adr = spec.joint_addr(model, a.prefix)
    kp, kd = spec.gains()
    default = spec.default_pose()
    # A scene can ask for a different resting elbow bend (spawn_pose.json
    # "wbc_default_elbow"), the same override G1_DEFAULT_ELBOW gives the
    # deployed controller: the stance, the arm targets and the balance net's
    # reference pose all move with it.
    spawn_cfg = Path(a.mjcf).resolve().parent.parent / "spawn_pose.json"
    if spawn_cfg.exists():
        elbow = json.loads(spawn_cfg.read_text()).get("wbc_default_elbow")
        if elbow is not None:
            for side in ("left", "right"):
                i = next(k for k, n in enumerate(spec.body_joints) if n.startswith(f"{side}_elbow"))
                default[i] = float(elbow)
            print(f"[wbc] resting elbow {float(elbow):+.2f} rad (from {spawn_cfg.name})")

    def stand():
        """Put the robot in the controller's default stance, feet on the floor."""
        mujoco.mj_resetData(model, data)
        data.qpos[adr["body_q"]] = default
        data.qpos[adr["hand_q"]] = 0.0
        mujoco.mj_forward(model, data)
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        floor_z = float(model.geom_pos[gid][2]) if gid >= 0 else 0.0
        robot_bodies = {int(model.jnt_bodyid[j]) for j in range(model.njnt)}
        low = float("inf")
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) not in robot_bodies:
                continue
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                   int(model.geom_bodyid[g])) or ""
            if a.prefix and not nm.startswith(a.prefix):
                continue
            pos, mat = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3)
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = int(model.geom_dataid[g])
                v0, n0 = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
                V = model.mesh_vert[v0:v0 + n0].reshape(-1, 3) @ mat.T + pos
                low = min(low, float(V[:, 2].min()))
            else:
                low = min(low, float(pos[2] - model.geom_size[g].max()))
        if np.isfinite(low):
            data.qpos[adr["base_q"] + 2] += floor_z + 0.002 - low
            mujoco.mj_forward(model, data)

    stand()

    # The viewer's backspace resets qpos to qpos0, which both drops the stance
    # and puts the objects back in one fixed layout.  Reshuffle them instead so
    # every episode starts somewhere new - within the robot's half of the table
    # and within arm's reach, or teleoperation gets awkward.
    rand = parker = None
    # Plate + fruits + basket scenes carry their own layout config next to the
    # scene's mujoco/ directory; everything else uses the apple/basket annulus.
    layout_cfg = Path(a.mjcf).resolve().parent.parent / "layout_random.json"
    if layout_cfg.exists():
        from randomize_plate_basket import PlateBasketRandomizer
        cfg = json.loads(layout_cfg.read_text())
        fruits = [f.strip() for f in a.fruits.split(",") if f.strip()]
        if fruits:
            cfg["only_fruits"] = fruits
        parker = PlateBasketRandomizer(model, data, cfg, seed=a.seed)
        rand = parker if a.randomize else None
    elif a.randomize:
        from randomize_objects import TableRandomizer
        try:
            rand = TableRandomizer(model, data, seed=a.seed,
                                   reach=(a.reach_min, a.reach_max))
        except ValueError as e:
            print(f"[wbc] object randomization off: {e}")

    def reset_layout():
        stand()
        if parker is not None:
            parker.park(data)        # left-out fruits stay off the table
            mujoco.mj_forward(model, data)
        if rand is not None:
            placed = rand.apply(data)
            mujoco.mj_forward(model, data)
            print("[wbc] layout " + "  ".join(
                f"{n.split('_', 1)[-1]}=({p[0]:+.3f},{p[1]:+.3f})"
                for n, p in placed.items()))

    reset_layout()
    print(f"[wbc] {spec.name}: stance set, base z={data.qpos[adr['base_q'] + 2]:+.3f}")

    wbc = Wbc(spec)
    wbc.default = default.copy()
    groot = None
    if a.policy_host:
        groot = Groot(a.policy_host, a.policy_port, a.lang)

    cam_name = a.camera
    if cam_name is None:
        if model.ncam == 0:
            raise SystemExit(f"{a.mjcf.name} defines no cameras - pass --camera")
        cam_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, 0)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        raise SystemExit(f"no camera '{cam_name}' in {a.mjcf.name}")
    fovy = a.fovy if a.fovy else float(model.cam_fovy[cam_id])
    f_px = (a.height / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)

    sensor = None
    if a.mode != "mujoco_only":
        if a.ply is None:
            raise SystemExit(f"--mode {a.mode} needs --ply")
        from mugs.sensors.gaussian_sensor import GaussianSensor, GaussianSensorConfig
        sensor = GaussianSensor(GaussianSensorConfig(
            width=a.width, height=a.height, background_ply_path=a.ply,
            render_mode=a.mode, robot_geom_ids=foreground_geom_ids(model, {"floor"}),
            use_sh=True, depth_composite=True, camera_convention="mujoco"))
    else:
        mesh_r = mujoco.Renderer(model, a.height, a.width)

    ego = ego_id = None
    if groot is not None:
        ego_name = a.ego_camera or f"{a.prefix}head_pinhole"
        ego_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, ego_name)
        if ego_id < 0:
            have = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                    for i in range(model.ncam)]
            raise SystemExit(f"no ego camera '{ego_name}'. Cameras: {have}")
        ego = mujoco.Renderer(model, a.ego_height, a.ego_width)

    passive = None
    if a.viewer:
        import mujoco.viewer
        passive = mujoco.viewer.launch_passive(model, data, show_left_ui=False,
                                               show_right_ui=False)
        print("[wbc] MuJoCo viewer attached to the live data")

    writer = None
    if a.record:
        writer = cv2.VideoWriter(str(a.record), cv2.VideoWriter_fourcc(*"mp4v"),
                                 int(a.policy_hz), (a.width, a.height))
    title = f"{spec.name} + WBC"
    if not a.no_window:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, a.width, a.height)

    ctrl_hz = float(spec.wbc["control_hz"])
    decim = max(1, round((1 / ctrl_hz) / model.opt.timestep))
    per_action = max(1, round(ctrl_hz / a.policy_hz))
    target = default.astype(float).copy()
    hand_target = np.zeros(spec.n_hand)
    nav = np.zeros(3, np.float32)
    height = float(spec.wbc["default_height"])
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    print(f"[wbc] control {ctrl_hz:.0f} Hz, render {a.policy_hz:.0f} Hz, "
          f"{decim} physics steps per control step")

    def state_dict():
        q, h = data.qpos[adr["body_q"]], data.qpos[adr["hand_q"]]
        nh = spec.n_hand // 2
        return {"left_leg": q[0:6], "right_leg": q[6:12], "waist": q[12:15],
                "left_arm": q[15:22], "right_arm": q[22:29],
                "left_hand": h[0:nh], "right_hand": h[nh:2 * nh]}

    pub = pub_hand_q = None
    if a.publish_state:
        import socket
        from wbc_state_relay import BODY_JOINTS as WIRE_BODY, HAND_JOINTS as WIRE_HAND

        # The wire format orders fingers thumb-first; a descriptor may order them
        # index-first (GR00T's hand-group order). Publishing in the descriptor's
        # order would curl the wrong fingers on the receiving side, so look the
        # wire names up explicitly rather than reusing adr["hand_q"].
        if list(WIRE_BODY) != list(spec.body_joints):
            raise SystemExit(
                "--publish-state: this robot's body_joints do not match the wire "
                "format in wbc_state_relay.py; extend that module before using it")
        import mujoco as _mj

        def _qadr(name):
            for cand in (f"{a.prefix}{name}_joint", f"{a.prefix}{name}"):
                j = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_JOINT, cand)
                if j >= 0:
                    return int(model.jnt_qposadr[j])
            raise SystemExit(f"--publish-state: no joint '{a.prefix}{name}'")

        pub_hand_q = np.array([_qadr(n) for n in WIRE_HAND])
        pub = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        pub_addr = ("127.0.0.1", a.publish_state)
        n_pkt = 1 + 3 + 4 + spec.n_body + len(pub_hand_q)
        print(f"[wbc] publishing state on udp {a.publish_state} "
              f"({n_pkt} float32 per datagram, wire finger order)")

    def publish(tick: int) -> None:
        if pub is None:
            return
        v = np.empty(1 + 3 + 4 + spec.n_body + len(pub_hand_q), np.float32)
        v[0] = tick
        v[1:4] = data.qpos[adr["base_q"]:adr["base_q"] + 3]
        v[4:8] = data.qpos[adr["base_q"] + 3:adr["base_q"] + 7]
        v[8:8 + spec.n_body] = data.qpos[adr["body_q"]]
        v[8 + spec.n_body:] = data.qpos[pub_hand_q]
        try:
            pub.sendto(v.tobytes(), pub_addr)
        except OSError:
            pass

    t0, step, queries = time.perf_counter(), 0, 0
    last_time = data.time
    try:
        while True:
            if groot is not None and step % per_action == 0:
                if groot.exhausted():
                    ego.update_scene(data, camera=ego_id)
                    groot.query(ego.render().copy(), state_dict())
                    queries += 1
                act = groot.current()
                groot.idx += 1
                target[15:22] = act["left_arm"]
                target[22:29] = act["right_arm"]
                nh = spec.n_hand // 2
                hand_target[0:nh] = act["left_hand"]
                hand_target[nh:2 * nh] = act["right_hand"]
                nav = np.asarray(act["navigate_command"], np.float32).reshape(3)
                height = float(np.asarray(act["base_height_command"]).reshape(-1)[0])

            target[:spec.n_act] = wbc.step(
                data.qpos[adr["body_q"]].astype(np.float32),
                data.qvel[adr["body_v"]].astype(np.float32),
                data.qpos[adr["base_q"] + 3:adr["base_q"] + 7].astype(np.float32),
                data.qvel[adr["base_v"] + 3:adr["base_v"] + 6].astype(np.float32),
                nav, height, np.zeros(3, np.float32))

            for _ in range(decim):
                tau = kp * (target - data.qpos[adr["body_q"]]) - kd * data.qvel[adr["body_v"]]
                data.ctrl[adr["body_a"]] = np.clip(tau, lo[adr["body_a"]], hi[adr["body_a"]])
                htau = (float(spec.wbc["hand_kp"]) * (hand_target - data.qpos[adr["hand_q"]])
                        - float(spec.wbc["hand_kd"]) * data.qvel[adr["hand_v"]])
                data.ctrl[adr["hand_a"]] = np.clip(htau, lo[adr["hand_a"]], hi[adr["hand_a"]])
                mujoco.mj_step(model, data)
            step += 1

            publish(step)

            if passive is not None:
                if not passive.is_running():
                    break
                if data.time < last_time - 1e-9:   # backspace in the viewer
                    reset_layout()
                    step = 0
                last_time = data.time
                passive.sync()

            if step % per_action == 0:
                if sensor is not None:
                    rgb = sensor.render(model, data, cam_name, camera_params={
                        "position": data.cam_xpos[cam_id].copy(),
                        "rotation": data.cam_xmat[cam_id].reshape(3, 3).copy(),
                        "fx": f_px, "fy": f_px,
                        "width": a.width, "height": a.height})
                else:
                    mesh_r.update_scene(data, camera=cam_id)
                    rgb = mesh_r.render()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                tag = "WBC only" if groot is None else "policy + WBC"
                cv2.putText(bgr, f"{tag}  nav {nav[0]:+.2f} {nav[1]:+.2f} {nav[2]:+.2f}"
                                 f"  h {height:.2f}"
                                 f"  base {data.qpos[adr['base_q'] + 2]:+.3f}  q{queries}",
                            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
                if writer is not None:
                    writer.write(bgr)
                if not a.no_window:
                    cv2.imshow(title, bgr)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        break
            if a.seconds and time.perf_counter() - t0 > a.seconds:
                break
            if not np.isfinite(data.qpos).all():
                print("[wbc] simulation diverged")
                break
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
            print(f"[wbc] wrote {a.record}")
        cv2.destroyAllWindows()
        if passive is not None:
            passive.close()
        if sensor is not None:
            sensor.close()
    print(f"[wbc] {step} control steps, {queries} policy queries, "
          f"base z={data.qpos[adr['base_q'] + 2]:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
