#!/usr/bin/env python3
"""Relay the gear_sonic sim's robot state to the hybrid viewer over UDP.

The photoreal viewer needs gsplat; the DDS client library needs cyclonedds. No
single environment here has both - robosnap-sim has gsplat and no unitree_sdk2py,
.venv_sim has unitree_sdk2py and no gsplat. So this runs in .venv_sim, listens to
the same topics the WBC listens to, and forwards the state as fixed-size UDP
datagrams that the viewer reads with nothing but the standard library.

Nothing is published: this is a pure observer, invisible to the sim and the WBC.

    $RS_PY_DDS \
        scripts/wbc_state_relay.py

One datagram = 51 float32 = [tick, base_pos(3), base_quat_wxyz(4),
body_q(29), hand_q(14)], joints in MuJoCo model order.

The robot is not the whole scene. Anything with a free joint - the apple, the
basket - is moved by the sim's physics too, and DDS carries none of it: rt/lowstate
is motor state, rt/odostate is the base. So the sim publishes a SECOND kind of
datagram on the same ports, tagged with a magic word, and the receiver below
writes both. Without it the renderer's apple stays at its spawn pose while the
sim's rolls off the table, and every photoreal frame - the hybrid, the robot's
camera, anything a dataset records - disagrees with the physics it came from.

    object datagram = b"RSOB" + uint32 n + n * (char[40] name + 7 float32 pose)

Names are matched after dropping build_wbc_scene's "rs_" prefix, so the sim's
rs_obj1_red_apple_free lands on the renderer's obj1_red_apple_free.
"""
from __future__ import annotations

import argparse
import socket
import time

import numpy as np

N_BODY = 29
N_HAND = 7
PACKET_FLOATS = 1 + 3 + 4 + N_BODY + 2 * N_HAND

OBJ_MAGIC = b"RSOB"
# b"RSVR" + uint32 seed + uint32 episode: the visual-randomization seed, sent by
# run_mujoco_wbc_sim.py --visual-dr. Kept in step with visual_randomizer.py.
VIS_MAGIC = b"RSVR"
OBJ_NAME_BYTES = 40
OBJ_STRIDE = OBJ_NAME_BYTES + 7 * 4


def obj_key(name: str) -> str:
    """The name both sides agree on: build_wbc_scene prefixes ours with rs_."""
    return name[3:] if name.startswith("rs_") else name


def pack_objects(names, poses) -> bytes:
    """poses[i] = [x, y, z, qw, qx, qy, qz] of free joint names[i], sim frame."""
    import struct
    out = [OBJ_MAGIC, struct.pack("<I", len(names))]
    for nm, q in zip(names, poses):
        out.append(nm.encode("utf-8")[:OBJ_NAME_BYTES].ljust(OBJ_NAME_BYTES, b"\0"))
        out.append(np.asarray(q, np.float32).tobytes())
    return b"".join(out)


def unpack_objects(pkt: bytes):
    """-> [(key, pose)], or [] if the datagram is not one of ours."""
    import struct
    if len(pkt) < 8 or pkt[:4] != OBJ_MAGIC:
        return []
    (n,) = struct.unpack("<I", pkt[4:8])
    if len(pkt) != 8 + n * OBJ_STRIDE:
        return []
    out = []
    for i in range(n):
        o = 8 + i * OBJ_STRIDE
        nm = pkt[o:o + OBJ_NAME_BYTES].split(b"\0", 1)[0].decode("utf-8", "replace")
        out.append((obj_key(nm),
                    np.frombuffer(pkt, np.float32, 7, o + OBJ_NAME_BYTES)))
    return out

BODY_JOINTS = [
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
# Dex3 MOTOR order, which is not the tidy thumb-first order and is not the same
# on both hands. rt/dex3/<side>/state carries motor_state[0..6] in exactly this
# order, so listing them any other way silently writes each finger's angle onto
# a different finger - the hand still moves, it just moves wrongly, which is far
# harder to spot than a hand that does not move at all.
HAND_JOINTS = ([f"left_hand_{f}" for f in
                ("index_0", "index_1", "middle_0", "middle_1",
                 "thumb_0", "thumb_1", "thumb_2")]
               + [f"right_hand_{f}" for f in
                  ("middle_0", "middle_1", "index_0", "index_1",
                   "thumb_0", "thumb_1", "thumb_2")])


class MirrorReceiver:
    """Write the relayed sim state into a RoboSnap scene's MjData.

    The sim runs in its own frame - make_wbc_scene.py moved the reconstruction so
    the robot's stance became the origin - so the base pose is transformed back
    before it is written, which is what keeps the robot registered to the splat.

    Imports no DDS: this half lives in the rendering environment.
    """

    def __init__(self, model, scene_dir, prefix: str = "g1_", port: int = 5601,
                 frame: str = "auto", objects: bool = True):
        import socket
        import mujoco
        from make_wbc_scene import floor_height, robot_pose

        def qadr(name):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{name}_joint")
            if jid < 0:
                raise SystemExit(f"--mirror: no joint '{prefix}{name}_joint' in the scene")
            return int(model.jnt_qposadr[jid])

        self.jadr = np.array([qadr(n) for n in BODY_JOINTS + HAND_JOINTS])
        self.badr = qadr("floating_base")

        # The frame shift only exists when the sim runs a scene that
        # make_wbc_scene.py re-origined onto the robot's stance. When the sim and
        # the renderer load the SAME mjcf - which is the case for
        # a scene_g1.xml driven by run_mujoco_wbc_sim.py directly -
        # the two frames are already identical and any transform would move the
        # robot off its own splat.
        frame_json = scene_dir / "mujoco" / "wbc" / "scene_wbc.frame.json"
        swr = scene_dir / "mujoco" / "scene_with_robot.xml"
        if frame == "identity":
            at, yaw = np.zeros(3), 0.0
            self.origin = np.zeros(3)
            print("[mirror] identity frame: the publisher and this renderer "
                  "load the same scene", flush=True)
        elif frame_json.is_file():
            import json
            fj = json.loads(frame_json.read_text(encoding="utf-8"))
            at, yaw = np.asarray(fj["at"], float), float(fj["yaw"])
            self.origin = np.array([at[0], at[1], float(fj["floor_z"])])
            print(f"[mirror] undoing build_wbc_scene's re-origin: "
                  f"at ({at[0]:+.3f}, {at[1]:+.3f}) yaw "
                  f"{np.degrees(yaw):+.1f} deg", flush=True)
        elif swr.is_file():
            at, yaw = robot_pose(swr, prefix)
            self.origin = np.array([at[0], at[1],
                                    floor_height(scene_dir / "mujoco" / "scene.xml")])
        else:
            at, yaw = np.zeros(3), 0.0
            self.origin = np.zeros(3)
            print(f"[mirror] no {swr.name}: sim and renderer share one frame, "
                  f"using the identity transform", flush=True)
        self.rot = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                             [np.sin(yaw), np.cos(yaw), 0.0],
                             [0.0, 0.0, 1.0]])
        self.q_yaw = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        self.yaw = yaw

        # Every free joint that is not the robot's base is a scene object the sim
        # also owns: the apple, the basket. Keyed by the shared name so the
        # sender's joint order is irrelevant and an object either side lacks is
        # simply skipped.
        self.objects = {}
        if objects:
            for j in range(model.njnt):
                if int(model.jnt_type[j]) != mujoco.mjtJoint.mjJNT_FREE:
                    continue
                adr = int(model.jnt_qposadr[j])
                if adr == self.badr:
                    continue
                nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
                self.objects[obj_key(nm)] = adr
            if self.objects:
                print(f"[mirror] following {len(self.objects)} scene object(s): "
                      f"{', '.join(sorted(self.objects))}", flush=True)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.setblocking(False)
        self.port, self.got, self.got_obj = port, 0, 0
        self.missing = set()
        # (seed, episode) from the sim's RSVR datagram, for renderers doing
        # visual domain randomization - see scripts/realsim/visual_randomizer.py.
        # None until the sim sends one; stays None when it runs without --visual-dr.
        self.visual = None

    def _pos(self, p):
        return self.rot @ np.asarray(p, float) + self.origin

    def _quat(self, q):
        w0, x0, y0, z0 = self.q_yaw
        w1, x1, y1, z1 = q
        return [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ]

    def apply(self, data) -> bool:
        """Take the newest datagram of each kind, drop the backlog, write qpos.

        Two senders share this port - wbc_state_relay for the robot, the sim
        itself for the objects - so the drain keeps the last of each kind rather
        than the last packet seen. Returns whether a ROBOT packet landed, which
        is what wait() below is waiting for.
        """
        pkt = objpkt = None
        while True:
            try:
                d = self.sock.recv(65535)
            except BlockingIOError:
                break
            if d[:4] == OBJ_MAGIC:
                objpkt = d
            elif d[:4] == VIS_MAGIC and len(d) == 12:
                self.visual = tuple(np.frombuffer(d, "<u4", 2, 4).tolist())
            elif len(d) == PACKET_FLOATS * 4:
                pkt = d

        if objpkt is not None and self.objects:
            for key, pose in unpack_objects(objpkt):
                adr = self.objects.get(key)
                if adr is None:
                    if key not in self.missing:
                        self.missing.add(key)
                        print(f"[mirror] the sim moves '{key}', which this scene "
                              f"does not have - ignoring it", flush=True)
                    continue
                data.qpos[adr:adr + 3] = self._pos(pose[0:3])
                data.qpos[adr + 3:adr + 7] = self._quat(pose[3:7])
            self.got_obj += 1

        if pkt is None:
            return False
        v = np.frombuffer(pkt, np.float32)
        data.qpos[self.badr:self.badr + 3] = self._pos(v[1:4])
        data.qpos[self.badr + 3:self.badr + 7] = self._quat(v[4:8])
        data.qpos[self.jadr] = v[8:]
        data.qvel[:] = 0.0
        self.got += 1
        return True

    def wait(self, data, timeout: float = 15.0) -> bool:
        """Block until the first packet lands, so frame 0 is not the rest pose."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if self.apply(data):
                return True
            time.sleep(0.02)
        return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", type=int, default=99, help="DDS domain (sim yaml: 99)")
    p.add_argument("--interface", default="lo")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default="5601",
                   help="destination UDP port, or a comma-separated list. Give "
                        "one per consumer: two sockets cannot bind() the same "
                        "port, and the second dies with 'Address already in use'.")
    p.add_argument("--rate", type=float, default=50.0)
    a = p.parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, LowState_, OdoState_

    ChannelFactoryInitialize(a.domain, a.interface)

    state = {"body": np.zeros(N_BODY, np.float32),
             "left": np.zeros(N_HAND, np.float32),
             "right": np.zeros(N_HAND, np.float32),
             "pos": np.zeros(3, np.float32),
             "quat": np.array([1, 0, 0, 0], np.float32),
             "tick": 0.0,
             "seen": False}

    def on_low(msg):
        for i in range(N_BODY):
            state["body"][i] = msg.motor_state[i].q
        state["tick"] = msg.tick
        state["seen"] = True

    def on_odo(msg):
        # The sim fills these straight from floating_base_pose, so position is the
        # true world pose - not an estimate we would have to integrate.
        state["pos"][:] = msg.position[:3]
        state["quat"][:] = msg.orientation[:4]

    def on_hand(key):
        def handler(msg):
            for i in range(N_HAND):
                state[key][i] = msg.motor_state[i].q
        return handler

    subs = [ChannelSubscriber("rt/lowstate", LowState_),
            ChannelSubscriber("rt/odostate", OdoState_),
            ChannelSubscriber("rt/dex3/left/state", HandState_),
            ChannelSubscriber("rt/dex3/right/state", HandState_)]
    for sub, cb in zip(subs, [on_low, on_odo, on_hand("left"), on_hand("right")]):
        sub.Init(cb, 1)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ports = [int(x) for x in str(a.port).replace(" ", "").split(",") if x]
    dsts = [(a.host, q) for q in ports]
    print(f"[relay] domain {a.domain} on {a.interface} -> udp {a.host}:{ports} "
          f"at {a.rate:.0f} Hz")
    print("[relay] waiting for the sim's first LowState ...")

    dt = 1.0 / a.rate
    announced = False
    sent = 0
    try:
        while True:
            t0 = time.perf_counter()
            if state["seen"]:
                if not announced:
                    print("[relay] receiving; streaming to the viewer", flush=True)
                    announced = True
                pkt = np.concatenate([
                    np.array([state["tick"]], np.float32), state["pos"], state["quat"],
                    state["body"], state["left"], state["right"]]).astype(np.float32)
                assert pkt.size == PACKET_FLOATS
                for d in dsts:
                    sock.sendto(pkt.tobytes(), d)
                sent += 1
                if sent % (int(a.rate) * 10) == 0:
                    print(f"\r[relay] {sent} packets, base "
                          f"({state['pos'][0]:+.2f}, {state['pos'][1]:+.2f}, "
                          f"{state['pos'][2]:+.2f})", end="", flush=True)
            lag = dt - (time.perf_counter() - t0)
            if lag > 0:
                time.sleep(lag)
    except KeyboardInterrupt:
        print("\n[relay] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
