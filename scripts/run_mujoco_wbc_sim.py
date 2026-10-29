#!/usr/bin/env python3

"""
Run a reconstructed G1 MuJoCo scene with Unitree DDS + GR00T WBC.

Communication:

    WBC
      |
      | rt/lowcmd
      v
    UnitreeSdk2Bridge
      |
      | desired q/dq/tau/kp/kd
      v
    PD controller
      |
      | torque
      v
    MuJoCo
      |
      | rt/lowstate / rt/odostate / IMU
      v
    WBC

DDS:
    DOMAIN_ID = 99
    INTERFACE  = lo

Example:

    PYTHONPATH=$RS_DEPLOY_ROOT:$PYTHONPATH \
    python scripts/run_mujoco_wbc_sim.py \
        --scene-dir outputs/myroom \
        --mjcf outputs/myroom/mujoco/scene_g1.xml
"""

from __future__ import annotations

import math
import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

# The wire format lives next to its reader, in the relay module.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wbc_state_relay import (  # noqa: E402
    BODY_JOINTS,
    HAND_JOINTS,
    PACKET_FLOATS,
    VIS_MAGIC,
    pack_objects,
)

from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import (
    UnitreeSdk2Bridge,
)


# ============================================================
# IMPORTANT: fixed DDS domain requested by you
# ============================================================

DOMAIN_ID = 99
INTERFACE = "lo"


class MujocoWBCSim:
    """
    Standalone MuJoCo simulator with Unitree DDS WBC interface.

    This intentionally reuses Gear Sonic's UnitreeSdk2Bridge rather
    than implementing a new DDS protocol.
    """

    def __init__(
        self,
        mjcf_path: Path,
        scene_dir: Path,
        no_hold: bool = False,
        no_randomize: bool = False,
        seed: int | None = None,
        reach_min: float = 0.18,
        reach_max: float = 0.45,
        apple_reach_min: float = 0.16,
        apple_reach_max: float = 0.40,
        paused: bool = False,
        publish_objects: str = "",
        publish_state: str = "",
        visual_dr: bool = False,
        visual_seed: int | None = None,
        control_port: int = 0,
        fruits: list | None = None,
    ):
        self.mjcf_path = Path(mjcf_path)

        # Visual domain randomization. This process renders nothing - it only
        # owns the episode, so it picks one seed per reset and broadcasts it;
        # every renderer derives the same look from it (visual_randomizer.py).
        # Episode 0 / seed 0 is the nominal scene, until the first reset.
        self.visual_dr = bool(visual_dr)
        self.visual_rng = np.random.default_rng(visual_seed)
        self.visual_episode = 0
        self.visual_seed = 0
        self.pending_visual_seed = None     # b"RSSD": use this on the next reset

        # Reset requests over UDP (b"RSRS"), so an unattended collection run
        # can do what backspace does without anyone at the window.
        self.ctrl_sock = None
        if control_port:
            import socket
            self.ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.ctrl_sock.bind(("127.0.0.1", int(control_port)))
            self.ctrl_sock.setblocking(False)
            print(f"[sim] reset requests accepted on udp {control_port}")
        self.scene_dir = Path(scene_dir)

        self.paused = paused

        # ----------------------------------------------------
        # Load MuJoCo
        # ----------------------------------------------------

        print(f"[MuJoCo] loading: {self.mjcf_path}")

        self.model = mujoco.MjModel.from_xml_path(
            str(self.mjcf_path)
        )
        self.data = mujoco.MjData(self.model)

        print(
            f"[MuJoCo] bodies={self.model.nbody}, "
            f"actuators={self.model.nu}, "
            f"nq={self.model.nq}, "
            f"nv={self.model.nv}"
        )

        mujoco.mj_forward(self.model, self.data)

        # ----------------------------------------------------
        # Determine robot structure
        # ----------------------------------------------------

        self.qpos_offset = 7
        self.qvel_offset = 6

        self._check_model()

        # ----------------------------------------------------
        # DDS
        # ----------------------------------------------------

        print(
            f"[DDS] Initializing Unitree SDK2 "
            f"domain_id={DOMAIN_ID}, interface={INTERFACE}"
        )

        ChannelFactoryInitialize(
            DOMAIN_ID,
            INTERFACE,
        )

        # ----------------------------------------------------
        # Bridge configuration
        #
        # These values must correspond to the G1 used by WBC.
        # ----------------------------------------------------

        self.bridge_config = {
            "ROBOT_TYPE": "g1",
            "NUM_MOTORS": 29,
            "NUM_HAND_MOTORS": 7,
            "USE_SENSOR": False,
        }

        print("[DDS] Creating UnitreeSdk2Bridge")

        self.bridge = UnitreeSdk2Bridge(
            self.bridge_config
        )

        # ----------------------------------------------------
        # Latest action
        #
        # We keep the latest WBC command between DDS updates.
        # WBC is 50 Hz while MuJoCo can run much faster.
        # ----------------------------------------------------

        self.latest_action = np.zeros(
            29 + 7 + 7,
            dtype=np.float64,
        )

        self.have_wbc_command = False

        # ----------------------------------------------------
        # Object state publisher
        #
        # DDS carries the robot and nothing else, so the
        # photoreal renderers cannot see the apple move: they
        # do not step physics while mirroring, and their copy
        # of it stays at the spawn pose. Publish every free
        # joint that is not the base straight to them, on the
        # same UDP ports the relay uses.
        # ----------------------------------------------------

        self._init_object_publisher(publish_objects)

        # ----------------------------------------------------
        # Robot state publisher
        #
        # wbc_state_relay.py exists because the rendering env
        # cannot speak DDS. But this process CAN: it is the
        # DDS bridge, and mj_data is right here. Publishing the
        # same 51-float packet straight from it removes the
        # relay process entirely - one publisher instead of a
        # sim, a relay, and the orphan-relay failure mode where
        # two of them double every window's state.
        #
        # The relay stays for sims we do not own (gear_sonic's
        # run_sim_loop.py): it observes the wire instead of
        # needing a hook inside the simulator.
        # ----------------------------------------------------

        self._init_state_publisher(publish_state)

        # ----------------------------------------------------
        # Optional initial joint hold
        #
        # This is only used until WBC starts sending commands.
        # Once WBC commands arrive, the WBC commands take over.
        # ----------------------------------------------------

        self.initial_hold = None

        if not no_hold:
            self.initial_hold = self._get_initial_joint_positions()

        # ----------------------------------------------------
        # Spawn pose
        #
        # A scene can pin the joint angles the robot spawns with and returns to
        # on every reset: <scene>/spawn_pose.json {"joints": {name: rad}}. Use it
        # to match the pose the WBC holds at rest, or the arms jump on reset.
        # "lateral_jitter_m": d also slides the robot sideways (along its own
        # left, never toward or away from the table) by U(-d, d) on every reset.
        # ----------------------------------------------------

        self.spawn_pose = {}
        self.lateral_jitter = 0.0
        self.last_sideways = 0.0
        self.spawn_rng = np.random.default_rng(seed)
        sp = Path(scene_dir) / "spawn_pose.json" if scene_dir else None
        if sp is not None and sp.exists():
            cfg = json.loads(sp.read_text())
            for jn, v in cfg.get("joints", {}).items():
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid < 0:
                    raise ValueError(f"{sp}: no joint {jn!r}")
                self.spawn_pose[int(self.model.jnt_qposadr[jid])] = float(v)
            self.lateral_jitter = float(cfg.get("lateral_jitter_m", 0.0))
            fb = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
            self.base_adr = int(self.model.jnt_qposadr[fb])
            w, x, y, z = self.model.qpos0[self.base_adr + 3:self.base_adr + 7]
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            self.base_left = np.array([-np.sin(yaw), np.cos(yaw)])
            print(f"[sim] spawn pose from {sp}: {cfg.get('joints', {})}, "
                  f"sideways jitter +-{self.lateral_jitter*100:.1f} cm")
            self.apply_spawn_pose()

        # ----------------------------------------------------
        # Table-top layout randomization
        #
        # The viewer's backspace resets qpos to qpos0, which puts the apple and
        # the basket back in one fixed spot every time. Data collection wants
        # variety, so reshuffle them on each reset - but only inside the half of
        # the table facing the robot and inside arm's reach, or the layout
        # becomes awkward to teleoperate.
        # ----------------------------------------------------

        # Built even with --no-randomize: exact replay (RSPL placement) needs it
        # to know the table top. --no-randomize then only stops the reshuffle
        # on reset, so a backspace no longer moves the apple somewhere random
        # just before collection.py places it where the demo had it.
        self.randomizer = None
        self.randomize_on_reset = not no_randomize
        if True:
            import sys as _sys
            _here = str(Path(__file__).resolve().parent / "realsim")
            if _here not in _sys.path:
                _sys.path.insert(0, _here)
            pb_cfg = Path(scene_dir) / "layout_random.json" if scene_dir else None
            if pb_cfg is not None and pb_cfg.exists():
                # Plate + fruits + basket scenes (layout_random.json):
                # the plate and basket keep or swap their photo spots, jittered,
                # and the fruits are shuffled on the plate.
                from randomize_plate_basket import PlateBasketRandomizer
                pb = json.loads(pb_cfg.read_text())
                if fruits:
                    pb["only_fruits"] = fruits
                self.randomizer = PlateBasketRandomizer(self.model, self.data, pb, seed=seed)
                self.randomizer.park(self.data)
                mujoco.mj_forward(self.model, self.data)
                print(f"[sim] object randomization on ({pb_cfg.name}): "
                      f"{[i['name'] for i in self.randomizer.items]}")
                if self.randomize_on_reset:
                    self.randomize_layout()
            else:
                try:
                    from randomize_objects import TableRandomizer
                    self.randomizer = TableRandomizer(
                        self.model, self.data, seed=seed,
                        reach=(reach_min, reach_max),
                        reach_overrides={"apple": (apple_reach_min,
                                                   apple_reach_max)})
                    print(f"[sim] object randomization on: "
                          f"{[i['name'] for i in self.randomizer.items]} "
                          f"reach {reach_min:.2f}-{reach_max:.2f} m "
                          f"(apple {apple_reach_min:.2f}-{apple_reach_max:.2f} m)")
                    if self.randomize_on_reset:
                        self.randomize_layout()
                    else:
                        print("[sim] layout reshuffle off: objects only move when "
                              "placed (exact replay)")
                except Exception as exc:
                    self.randomizer = None
                    print(f"[sim] object randomization off: {exc}")

        # ----------------------------------------------------
        # Per-episode layout log
        #
        # <scene>/episode_layouts/<session>.jsonl: one "reset" line per episode
        # (layout drawn, robot offset, visual seed, every object's pose) and a
        # "poses" line every 0.5 s, so a recorded episode's start layout and end
        # state can be looked up by wall-clock time.
        # ----------------------------------------------------

        self.layout_log = None
        self.reset_count = 0
        self._next_pose_log = 0.0
        if scene_dir:
            d = Path(scene_dir) / "episode_layouts"
            d.mkdir(exist_ok=True)
            self.layout_log_path = d / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}.jsonl"
            self.layout_log = open(self.layout_log_path, "a", buffering=1)
            fb = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
            self._log_base = int(self.model.jnt_qposadr[fb]) if fb >= 0 else None
            self._log_objs = [
                (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                   int(self.model.jnt_bodyid[j])), int(self.model.jnt_qposadr[j]))
                for j in range(self.model.njnt)
                if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and j != fb]
            print(f"[sim] episode layout log: {self.layout_log_path}")
            self.log_reset("start")

        # ----------------------------------------------------
        # Timing
        # ----------------------------------------------------

        self.sim_dt = float(self.model.opt.timestep)

        print(f"[MuJoCo] timestep = {self.sim_dt:.6f} s")
        print(f"[MuJoCo] frequency = {1.0 / self.sim_dt:.1f} Hz")

    # ========================================================
    # Model validation
    # ========================================================

    def _check_model(self):
        """
        Check that the reconstructed model has the basic G1
        structure expected by the existing Gear Sonic bridge.
        """

        body_names = []

        for i in range(self.model.nbody):
            name = self.model.body(i).name

            if name:
                body_names.append(name)

        print("[MuJoCo] checking model...")

        # Floating base is expected by the Gear Sonic observation
        # convention.
        floating_base_found = False

        for i in range(self.model.njnt):
            name = self.model.joint(i).name

            if name == "floating_base_joint":
                floating_base_found = True

        if floating_base_found:
            print("[MuJoCo] floating_base_joint: OK")
        else:
            print(
                "[MuJoCo] WARNING: floating_base_joint not found."
            )
            print(
                "[MuJoCo] The simulator assumes a free/floating "
                "G1 base with 7 qpos + 6 qvel."
            )

        if "pelvis" in body_names:
            print("[MuJoCo] pelvis: OK")
        else:
            print("[MuJoCo] WARNING: pelvis body not found")

        if "torso_link" in body_names:
            print("[MuJoCo] torso_link: OK")
        else:
            print("[MuJoCo] WARNING: torso_link body not found")

        if self.model.nu < 29:
            raise RuntimeError(
                f"Your MuJoCo model has only {self.model.nu} "
                f"actuators, but the G1 WBC expects at least "
                f"29 body actuators."
            )

    # ========================================================
    # Initial hold
    # ========================================================

    def _get_initial_joint_positions(self):
        """
        Store the initial position for every actuator whose
        transmission is associated with a hinge/slide joint.
        """

        hold = np.zeros(
            self.model.nu,
            dtype=np.float64,
        )

        for actuator_id in range(self.model.nu):

            joint_id = int(
                self.model.actuator_trnid[
                    actuator_id, 0
                ]
            )

            if joint_id < 0:
                continue

            joint_type = int(
                self.model.jnt_type[joint_id]
            )

            if joint_type in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):

                qpos_address = int(
                    self.model.jnt_qposadr[joint_id]
                )

                hold[actuator_id] = (
                    self.data.qpos[qpos_address]
                )

        return hold

    # ========================================================
    # Joint mapping
    # ========================================================

    def _get_actuator_joint_qpos(self, actuator_id):
        """
        Return the qpos address associated with an actuator.
        """

        joint_id = int(
            self.model.actuator_trnid[
                actuator_id, 0
            ]
        )

        if joint_id < 0:
            return None

        return int(
            self.model.jnt_qposadr[joint_id]
        )

    def _get_actuator_joint_dof(self, actuator_id):
        """
        Return the qvel/dof address associated with an actuator.
        """

        joint_id = int(
            self.model.actuator_trnid[
                actuator_id, 0
            ]
        )

        if joint_id < 0:
            return None

        return int(
            self.model.jnt_dofadr[joint_id]
        )

    # ========================================================
    # WBC command -> MuJoCo torque
    # ========================================================

    def compute_body_torques(self):
        """
        Convert Unitree LowCmd body commands to PD torques.

        tau =
            tau_ff
            + kp * (q_des - q)
            + kd * (dq_des - dq)

        This follows the same command-side PD structure used
        by Gear Sonic's MuJoCo simulator.
        """

        torques = np.zeros(
            29,
            dtype=np.float64,
        )

        # No command yet.
        if not self.bridge.low_cmd_received:
            return torques

        for i in range(29):

            cmd = self.bridge.low_cmd.motor_cmd[i]

            # ------------------------------------------------
            # IMPORTANT:
            #
            # The WBC's G1 body joint order is assumed to match
            # the first 29 actuators in the reconstructed MJCF.
            #
            # If your Real2Sim model uses a different ordering,
            # we will add an explicit name-based mapping.
            # ------------------------------------------------

            actuator_id = i

            if actuator_id >= self.model.nu:
                continue

            qpos_addr = self._get_actuator_joint_qpos(
                actuator_id
            )

            qvel_addr = self._get_actuator_joint_dof(
                actuator_id
            )

            if qpos_addr is None or qvel_addr is None:
                continue

            q = self.data.qpos[qpos_addr]
            dq = self.data.qvel[qvel_addr]

            q_des = float(cmd.q)
            dq_des = float(cmd.dq)

            kp = float(cmd.kp)
            kd = float(cmd.kd)

            tau_ff = float(cmd.tau)

            torques[i] = (
                tau_ff
                + kp * (q_des - q)
                + kd * (dq_des - dq)
            )

        return torques

    # ========================================================
    # Hand torque
    # ========================================================

    def compute_hand_torques(self):
        """
        Compute left + right hand PD torques.

        This follows Gear Sonic's existing implementation.
        """

        left = np.zeros(
            7,
            dtype=np.float64,
        )

        right = np.zeros(
            7,
            dtype=np.float64,
        )

        if not self.bridge.low_cmd_received:
            return np.concatenate(
                [left, right]
            )

        # ----------------------------------------------------
        # Left hand
        # ----------------------------------------------------

        for i in range(7):

            cmd = (
                self.bridge
                .left_hand_cmd
                .motor_cmd[i]
            )

            actuator_id = 29 + i

            if actuator_id >= self.model.nu:
                continue

            qpos_addr = self._get_actuator_joint_qpos(
                actuator_id
            )

            qvel_addr = self._get_actuator_joint_dof(
                actuator_id
            )

            if qpos_addr is None or qvel_addr is None:
                continue

            q = self.data.qpos[qpos_addr]
            dq = self.data.qvel[qvel_addr]

            left[i] = (
                float(cmd.tau)
                + float(cmd.kp)
                * (float(cmd.q) - q)
                + float(cmd.kd)
                * (float(cmd.dq) - dq)
            )

        # ----------------------------------------------------
        # Right hand
        # ----------------------------------------------------

        for i in range(7):

            cmd = (
                self.bridge
                .right_hand_cmd
                .motor_cmd[i]
            )

            actuator_id = 29 + 7 + i

            if actuator_id >= self.model.nu:
                continue

            qpos_addr = self._get_actuator_joint_qpos(
                actuator_id
            )

            qvel_addr = self._get_actuator_joint_dof(
                actuator_id
            )

            if qpos_addr is None or qvel_addr is None:
                continue

            q = self.data.qpos[qpos_addr]
            dq = self.data.qvel[qvel_addr]

            right[i] = (
                float(cmd.tau)
                + float(cmd.kp)
                * (float(cmd.q) - q)
                + float(cmd.kd)
                * (float(cmd.dq) - dq)
            )

        return np.concatenate(
            [left, right]
        )

    # ========================================================
    # Observation generation
    # ========================================================

    def prepare_obs(self):
        """
        Convert MuJoCo state into the observation dictionary
        expected by UnitreeSdk2Bridge.PublishLowState().
        """

        obs = {}

        # ----------------------------------------------------
        # Floating base
        # ----------------------------------------------------

        if self.model.nq >= 7 and self.model.nv >= 6:

            obs["floating_base_pose"] = (
                self.data.qpos[:7].copy()
            )

            obs["floating_base_vel"] = (
                self.data.qvel[:6].copy()
            )

            obs["floating_base_acc"] = (
                self.data.qacc[:6].copy()
            )

        else:

            obs["floating_base_pose"] = np.zeros(
                7,
                dtype=np.float64,
            )

            obs["floating_base_vel"] = np.zeros(
                6,
                dtype=np.float64,
            )

            obs["floating_base_acc"] = np.zeros(
                6,
                dtype=np.float64,
            )

        # ----------------------------------------------------
        # Torso IMU
        # ----------------------------------------------------

        torso_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "torso_link",
        )

        if torso_id >= 0:

            obs["secondary_imu_quat"] = (
                self.data.xquat[torso_id].copy()
            )

            pose = np.zeros(
                13,
                dtype=np.float64,
            )

            mujoco.mj_objectVelocity(
                self.model,
                self.data,
                mujoco.mjtObj.mjOBJ_BODY,
                torso_id,
                pose[7:13],
                1,
            )

            # MuJoCo returns:
            #
            # [angular velocity, linear velocity]
            #
            # Gear Sonic expects:
            #
            # [linear velocity, angular velocity]

            pose[7:10], pose[10:13] = (
                pose[10:13].copy(),
                pose[7:10].copy(),
            )

            obs["secondary_imu_vel"] = (
                pose[7:13].copy()
            )

        else:

            obs["secondary_imu_quat"] = np.array(
                [1.0, 0.0, 0.0, 0.0],
                dtype=np.float64,
            )

            obs["secondary_imu_vel"] = np.zeros(
                6,
                dtype=np.float64,
            )

        # ----------------------------------------------------
        # Body joints
        #
        # Assumes first 29 actuators correspond to the
        # 29 G1 body joints.
        # ----------------------------------------------------

        body_q = np.zeros(
            29,
            dtype=np.float64,
        )

        body_dq = np.zeros(
            29,
            dtype=np.float64,
        )

        body_ddq = np.zeros(
            29,
            dtype=np.float64,
        )

        body_tau = np.zeros(
            29,
            dtype=np.float64,
        )

        for i in range(29):

            if i >= self.model.nu:
                continue

            qpos_addr = self._get_actuator_joint_qpos(i)
            qvel_addr = self._get_actuator_joint_dof(i)

            if qpos_addr is None or qvel_addr is None:
                continue

            body_q[i] = self.data.qpos[qpos_addr]
            body_dq[i] = self.data.qvel[qvel_addr]
            body_ddq[i] = self.data.qacc[qvel_addr]

            # actuator_force is indexed by actuator
            body_tau[i] = self.data.actuator_force[i]

        obs["body_q"] = body_q
        obs["body_dq"] = body_dq
        obs["body_ddq"] = body_ddq
        obs["body_tau_est"] = body_tau

        # ----------------------------------------------------
        # Hands
        # ----------------------------------------------------

        left_q = np.zeros(
            7,
            dtype=np.float64,
        )

        left_dq = np.zeros(
            7,
            dtype=np.float64,
        )

        left_ddq = np.zeros(
            7,
            dtype=np.float64,
        )

        left_tau = np.zeros(
            7,
            dtype=np.float64,
        )

        right_q = np.zeros(
            7,
            dtype=np.float64,
        )

        right_dq = np.zeros(
            7,
            dtype=np.float64,
        )

        right_ddq = np.zeros(
            7,
            dtype=np.float64,
        )

        right_tau = np.zeros(
            7,
            dtype=np.float64,
        )

        # Left hand
        for i in range(7):

            actuator_id = 29 + i

            if actuator_id >= self.model.nu:
                continue

            qpos_addr = self._get_actuator_joint_qpos(
                actuator_id
            )

            qvel_addr = self._get_actuator_joint_dof(
                actuator_id
            )

            if qpos_addr is None or qvel_addr is None:
                continue

            left_q[i] = self.data.qpos[qpos_addr]
            left_dq[i] = self.data.qvel[qvel_addr]
            left_ddq[i] = self.data.qacc[qvel_addr]
            left_tau[i] = self.data.actuator_force[
                actuator_id
            ]

        # Right hand
        for i in range(7):

            actuator_id = 29 + 7 + i

            if actuator_id >= self.model.nu:
                continue

            qpos_addr = self._get_actuator_joint_qpos(
                actuator_id
            )

            qvel_addr = self._get_actuator_joint_dof(
                actuator_id
            )

            if qpos_addr is None or qvel_addr is None:
                continue

            right_q[i] = self.data.qpos[qpos_addr]
            right_dq[i] = self.data.qvel[qvel_addr]
            right_ddq[i] = self.data.qacc[qvel_addr]
            right_tau[i] = self.data.actuator_force[
                actuator_id
            ]

        obs["left_hand_q"] = left_q
        obs["left_hand_dq"] = left_dq
        obs["left_hand_ddq"] = left_ddq
        obs["left_hand_tau_est"] = left_tau

        obs["right_hand_q"] = right_q
        obs["right_hand_dq"] = right_dq
        obs["right_hand_ddq"] = right_ddq
        obs["right_hand_tau_est"] = right_tau

        obs["time"] = float(
            self.data.time
        )

        return obs

    # ========================================================
    # Receive WBC command
    # ========================================================

    def update_wbc_command(self):
        """
        Get the newest command from DDS.

        The bridge's callbacks run asynchronously. We keep the
        latest command and reuse it until another command arrives.
        """

        (
            action,
            cmd_received,
            is_new_action,
        ) = self.bridge.GetAction()

        if cmd_received:

            self.latest_action = np.asarray(
                action,
                dtype=np.float64,
            )

            self.have_wbc_command = True

        return is_new_action

    # ========================================================
    # Apply WBC command
    # ========================================================

    def apply_wbc_torques(self):

        if not self.have_wbc_command:

            # Before WBC starts, optionally hold the initial pose.
            if self.initial_hold is not None:

                self.data.ctrl[:] = (
                    self.initial_hold
                )

            return

        # ----------------------------------------------------
        # Body
        # ----------------------------------------------------

        body_torques = (
            self.compute_body_torques()
        )

        # ----------------------------------------------------
        # Hands
        # ----------------------------------------------------

        hand_torques = (
            self.compute_hand_torques()
        )

        # ----------------------------------------------------
        # Apply to MuJoCo
        #
        # IMPORTANT:
        # This assumes actuator ordering:
        #
        #   0..28    body
        #   29..35   left hand
        #   36..42   right hand
        #
        # We will verify this against your MJCF.
        # ----------------------------------------------------

        n_body = min(
            29,
            self.model.nu,
        )

        self.data.ctrl[:n_body] = (
            body_torques[:n_body]
        )

        # Left hand
        for i in range(7):

            actuator_id = 29 + i

            if actuator_id < self.model.nu:

                self.data.ctrl[actuator_id] = (
                    hand_torques[i]
                )

        # Right hand
        for i in range(7):

            actuator_id = 29 + 7 + i

            if actuator_id < self.model.nu:

                self.data.ctrl[actuator_id] = (
                    hand_torques[7 + i]
                )

        # ----------------------------------------------------
        # Torque clipping
        #
        # Do NOT add arbitrary limits here yet.
        #
        # The reconstructed MJCF / actuator definitions should
        # define the actual limits.
        # ----------------------------------------------------

    # ========================================================
    # Main simulation step
    # ========================================================

    def step(self):

        # ----------------------------------------------------
        # 1. Read WBC command
        # ----------------------------------------------------

        self.update_wbc_command()

        # ----------------------------------------------------
        # 2. Publish MuJoCo state to WBC
        # ----------------------------------------------------

        obs = self.prepare_obs()

        self.bridge.PublishLowState(
            obs
        )

        # ----------------------------------------------------
        # 3. Apply latest WBC command
        # ----------------------------------------------------

        self.apply_wbc_torques()

        # ----------------------------------------------------
        # 4. Step MuJoCo
        # ----------------------------------------------------

        mujoco.mj_step(
            self.model,
            self.data,
        )

    # ========================================================
    # Object state publisher
    # ========================================================

    def _init_object_publisher(self, ports: str):

        self.obj_dsts = []
        self.obj_names = []
        self.obj_adr = []
        self.obj_next = 0.0

        if not ports:
            return

        import socket

        base_adr = None

        for j in range(self.model.njnt):

            if int(self.model.jnt_type[j]) != mujoco.mjtJoint.mjJNT_FREE:
                continue

            name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                j,
            ) or ""

            body = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(self.model.jnt_bodyid[j]),
            ) or ""

            # The robot's own base goes out over DDS as rt/odostate; sending it
            # here as well would have two writers fighting over one qpos slot.
            if "floating_base" in name or "pelvis" in body:
                base_adr = int(self.model.jnt_qposadr[j])
                continue

            self.obj_names.append(name)
            self.obj_adr.append(int(self.model.jnt_qposadr[j]))

        self.obj_sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        self.obj_dsts = [
            ("127.0.0.1", int(q))
            for q in str(ports).replace(" ", "").split(",")
            if q
        ]

        print(
            f"[objects] publishing {len(self.obj_names)} free joint(s) to udp "
            f"{[d[1] for d in self.obj_dsts]}: "
            f"{', '.join(self.obj_names) if self.obj_names else 'none'}"
        )

        if base_adr is None:
            print(
                "[objects] WARNING: no floating base found, so nothing was "
                "excluded - check the scene"
            )

        if not self.obj_names:
            print(
                "[objects] nothing to publish: this scene has no free joint "
                "besides the base, so its objects are static geometry"
            )

    def _init_state_publisher(self, ports: str):

        self.state_dsts = []
        self.state_next = 0.0

        if not ports:
            return

        import socket

        def qadr(name):
            jid = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{name}_joint",
            )
            if jid < 0:
                raise SystemExit(
                    f"--publish-state: no joint '{name}_joint' in this scene. "
                    f"The wire format is fixed by wbc_state_relay.py; a scene "
                    f"missing one of its 43 joints cannot fill the packet."
                )
            return int(self.model.jnt_qposadr[jid])

        self.state_jadr = np.array(
            [qadr(n) for n in BODY_JOINTS + HAND_JOINTS]
        )

        self.state_badr = qadr("floating_base")

        self.state_sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        self.state_dsts = [
            ("127.0.0.1", int(q))
            for q in str(ports).replace(" ", "").split(",")
            if q
        ]

        print(
            f"[state] publishing the robot ({len(self.state_jadr)} joints + "
            f"base) to udp {[d[1] for d in self.state_dsts]} - "
            f"wbc_state_relay.py is not needed"
        )

    def publish_state(self, rate: float = 50.0):
        """Send the 51-float packet wbc_state_relay.py would have sent."""

        if not self.state_dsts:
            return

        now = time.time()

        if now < self.state_next:
            return

        self.state_next = now + 1.0 / rate

        q = self.data.qpos

        pkt = np.concatenate(
            [
                np.array([self.data.time], np.float32),
                q[self.state_badr:self.state_badr + 3],       # base pos
                q[self.state_badr + 3:self.state_badr + 7],   # base quat wxyz
                q[self.state_jadr],                           # 29 body + 14 hand
            ]
        ).astype(np.float32)

        assert pkt.size == PACKET_FLOATS, (
            f"packet is {pkt.size} floats, the wire format is {PACKET_FLOATS}"
        )

        for d in self.state_dsts:
            self.state_sock.sendto(pkt.tobytes(), d)

    def _match_published_framing(self, viewer):
        """Show, through a named camera, what that camera actually publishes.

        The viewer draws a fixed camera at the WINDOW's aspect ratio and ignores
        the camera's `resolution`. fovy is vertical and is honoured, so the
        vertical extent is right but the horizontal extent is whatever the
        window happens to be - usually far wider than the published frame. That
        is why the viewer and the image-view windows disagree for the same
        camera.

        Rather than change the calibrated fovy in the MJCF, re-derive it here
        for the live window so the HORIZONTAL field matches the published
        stream. Every named camera in these scenes is 16:9 (768x432, 1280x720),
        so at a 16:9 window this is a no-op and both axes match exactly; at any
        other shape the horizontal still matches and the vertical gives.
        """
        ui = getattr(self, "_ui", None)
        if ui is None:
            return
        if not hasattr(self, "_cam_fovy0"):
            self._cam_fovy0 = self.model.cam_fovy.copy()
        try:
            vp = ui.viewport
            win_aspect = float(vp.width) / float(vp.height)
            cam = viewer.cam
            cid = int(cam.fixedcamid)
            fixed = int(cam.type) == int(mujoco.mjtCamera.mjCAMERA_FIXED)
        except Exception:
            return
        if win_aspect <= 0:
            return

        # Anything not currently selected goes back to its calibrated value, so
        # a stale override cannot leak into the next camera or onto the free one.
        for i in range(self.model.ncam):
            if not (fixed and i == cid):
                self.model.cam_fovy[i] = self._cam_fovy0[i]
        if not fixed or not (0 <= cid < self.model.ncam):
            return

        res = self.model.cam_resolution[cid]
        if res[0] <= 0 or res[1] <= 0:
            return                      # no declared frame: nothing to match
        cam_aspect = float(res[0]) / float(res[1])
        fovy0 = float(self._cam_fovy0[cid])
        hfov = 2.0 * math.atan(math.tan(math.radians(fovy0) / 2.0) * cam_aspect)
        fovy = 2.0 * math.degrees(math.atan(math.tan(hfov / 2.0) / win_aspect))
        self.model.cam_fovy[cid] = fovy

    def new_visual_episode(self):
        """Draw the next visual seed. Called on every reset, layout or not."""
        if not self.visual_dr:
            return
        self.visual_episode += 1
        if self.pending_visual_seed is not None:
            # Asked for a specific look - redoing an episode with its lighting.
            self.visual_seed, self.pending_visual_seed = self.pending_visual_seed, None
        else:
            self.visual_seed = int(self.visual_rng.integers(1, 2**32 - 1))
        print(f"[sim] visual episode {self.visual_episode} "
              f"seed {self.visual_seed}", flush=True)
        self.obj_next = 0.0                 # send it now, not at the next tick

    def poll_control(self):
        """Drain the control socket -> (reset requested?, placements or None).

        b"RSRS"                            reset, exactly what backspace does
        b"RSSD" + uint32 seed              use this visual seed on the next reset
        b"RSPL" + uint32 n + n * (char[16] name needle, float32 x, y, yaw)
                                           put those objects there, sim frame,
                                           resting on the table - exact replay
                                           of a recorded demo's layout
        """
        if self.ctrl_sock is None:
            return False, None
        hit, place = False, None
        while True:
            try:
                d = self.ctrl_sock.recv(4096)
            except BlockingIOError:
                return hit, place
            if d[:4] == b"RSRS":
                hit = True
            elif d[:4] == b"RSSD" and len(d) == 8:
                self.pending_visual_seed = int(np.frombuffer(d, "<u4", 1, 4)[0])
                print(f"[sim] next reset will use visual seed "
                      f"{self.pending_visual_seed}", flush=True)
            elif d[:4] == b"RSPL" and len(d) >= 8:
                n = int(np.frombuffer(d, "<u4", 1, 4)[0])
                if len(d) == 8 + n * 28:
                    place = []
                    for i in range(n):
                        o = 8 + i * 28
                        name = d[o:o + 16].split(b"\0", 1)[0].decode()
                        x, y, yaw = np.frombuffer(d, "<f4", 3, o + 16)
                        place.append((name, float(x), float(y), float(yaw)))

    def place_objects(self, place):
        """Set objects to requested xy/yaw, resting on the table top."""
        if self.randomizer is None:
            print("[sim] cannot place objects: layout randomizer is off "
                  "(it is what knows the table top); start without "
                  "--no-randomize", flush=True)
            return
        out = []
        for needle, x, y, yaw in place:
            it = next((i for i in self.randomizer.items if needle in i["name"]), None)
            if it is None:
                print(f"[sim] place: no object matching {needle!r}", flush=True)
                continue
            q = it["quat"]
            # `yaw` is the world heading wanted for the object's body x axis;
            # turn it into a rotation off the nominal pose. NaN = keep nominal.
            if np.isnan(yaw):
                yaw = 0.0
            else:
                w_, x_, y_, z_ = q
                nominal = np.arctan2(2 * (x_ * y_ + w_ * z_),
                                     1 - 2 * (y_ * y_ + z_ * z_))
                yaw = yaw - nominal
            c, s = np.cos(yaw / 2), np.sin(yaw / 2)
            spun = np.array([c * q[0] - s * q[3], c * q[1] - s * q[2],
                             c * q[2] + s * q[1], c * q[3] + s * q[0]])
            a = it["adr"]
            self.data.qpos[a:a + 2] = (x, y)
            self.data.qpos[a + 2] = self.randomizer.top_z + it["lift"] + 0.002
            self.data.qpos[a + 3:a + 7] = spun / np.linalg.norm(spun)
            out.append(f"{needle}=({x:+.3f},{y:+.3f})")
        self.randomizer.zero_velocities(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        print("[sim] placed " + "  ".join(out), flush=True)

    def _object_poses(self):
        q = self.data.qpos
        # [x, y, z, qw, qx, qy, qz] per free body, world frame
        out = {n: np.round(q[a:a + 7], 4).tolist() for n, a in self._log_objs}
        if self._log_base is not None:
            out["robot_base"] = np.round(q[self._log_base:self._log_base + 7], 4).tolist()
        return out

    def log_reset(self, cause):
        if self.layout_log is None:
            return
        self.reset_count += 1
        rz = self.randomizer
        rec = {"t": round(time.time(), 3), "type": "reset", "reset": self.reset_count,
               "cause": str(cause), "visual_seed": int(self.visual_seed),
               "robot_sideways_m": round(self.last_sideways, 4),
               "layout": getattr(rz, "last", None),
               "fruits": [i["name"] for i in getattr(rz, "items", [])[2:]]
               if hasattr(rz, "park") else None,
               "poses": self._object_poses()}
        self.layout_log.write(json.dumps(rec) + "\n")

    def log_poses(self, period=0.5):
        if self.layout_log is None or time.time() < self._next_pose_log:
            return
        self._next_pose_log = time.time() + period
        self.layout_log.write(json.dumps({"t": round(time.time(), 3), "type": "poses",
                                          "poses": self._object_poses()}) + "\n")

    def apply_spawn_pose(self):
        for adr, v in self.spawn_pose.items():
            self.data.qpos[adr] = v
        self.last_sideways = 0.0
        if self.lateral_jitter > 0:
            off = self.spawn_rng.uniform(-self.lateral_jitter, self.lateral_jitter)
            self.last_sideways = float(off)
            a = self.base_adr
            self.data.qpos[a:a + 2] = self.model.qpos0[a:a + 2] + self.base_left * off
            print(f"[sim] robot spawn {off*100:+.1f} cm sideways "
                  f"({'left' if off > 0 else 'right'})", flush=True)
        mujoco.mj_forward(self.model, self.data)

    def randomize_layout(self):
        """Give the table-top objects a fresh, reachable layout."""
        if self.randomizer is not None and hasattr(self.randomizer, "park"):
            self.randomizer.park(self.data)       # left-out fruits stay away
            mujoco.mj_forward(self.model, self.data)
        if self.randomizer is None or not self.randomize_on_reset:
            return
        placed = self.randomizer.apply(self.data)
        self.randomizer.zero_velocities(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if placed:
            print("[sim] layout " + "  ".join(
                f"{n.split('_', 1)[-1]}=({p[0]:+.3f},{p[1]:+.3f})"
                for n, p in placed.items()), flush=True)

    def publish_objects(self, rate: float = 50.0):
        """Send the current pose of every tracked free joint, rate-limited."""

        if not self.obj_dsts or not self.obj_names:
            return

        now = time.time()

        if now < self.obj_next:
            return

        self.obj_next = now + 1.0 / rate

        poses = [
            self.data.qpos[a:a + 7]
            for a in self.obj_adr
        ]

        pkt = pack_objects(self.obj_names, poses)

        for d in self.obj_dsts:
            self.obj_sock.sendto(pkt, d)

        # Repeated at the object rate rather than sent once, so a renderer
        # started (or restarted) mid-episode still picks up the current look.
        if self.visual_dr:
            vis = VIS_MAGIC + np.array([self.visual_seed, self.visual_episode],
                                       "<u4").tobytes()
            for d in self.obj_dsts:
                self.obj_sock.sendto(vis, d)

    # ========================================================
    # Run
    # ========================================================

    def run(self):

        print()
        print("=" * 70)
        print("MuJoCo + Unitree DDS + WBC")
        print("=" * 70)
        print(f"scene       : {self.mjcf_path}")
        print(f"domain ID   : {DOMAIN_ID}")
        print(f"interface   : {INTERFACE}")
        print(f"sim dt      : {self.sim_dt}")
        print(f"sim Hz      : {1.0 / self.sim_dt:.1f}")
        print("=" * 70)
        print()

        print(
            "[WBC] waiting for rt/lowcmd..."
        )

        # Why the viewer's pause button used to do nothing.
        #
        # launch_passive means THIS loop owns mj_step - the viewer only draws.
        # Its Pause button and SPACE toggle the viewer's OWN run flag, which in
        # passive mode has no physics loop attached, so the robot kept stepping
        # regardless. Measured: SPACE never even reached a key_callback, because
        # the built-in UI consumes its own bindings before the callback runs.
        #
        # The fix is to read the flag the button actually writes. Handle._get_sim()
        # exposes the underlying Simulate, whose `run` is exactly that state, so
        # the native button and SPACE now drive this loop and the UI checkbox
        # always reflects reality.
        #
        # SIGUSR1 stays as the no-terminal, no-window route, and writes the same
        # flag so the UI does not disagree with us:
        #   kill -USR1 $(pgrep -f run_mujoco_wbc_sim)
        self._running = not self.paused
        self._single_step = False
        self._ui = None

        def key_callback(keycode):
            # Only reached for keys the viewer's UI does not claim itself.
            if keycode in (46, 190) and not self._running:   # '.' single step
                self._single_step = True

        import signal

        # The UI's flag is read-only, so SIGUSR1 cannot write it. Instead it
        # flips an inversion: effective = ui.run XOR invert. Both controls then
        # work independently and neither can leave the sim with no way to pause
        # - which matters because whether keystrokes reach the viewer at all
        # depends on the window manager.
        # --paused starts here rather than by writing the UI's flag, which is
        # read-only. The button then still works: pressing it flips ui.run, the
        # XOR flips with it, and the sim runs.
        self._invert = bool(self.paused)

        def on_usr1(_sig, _frm):
            self._invert = not self._invert
            self._running = not self._running
            print(f"[sim] {'running' if self._running else 'PAUSED'} (SIGUSR1)",
                  flush=True)

        signal.signal(signal.SIGUSR1, on_usr1)

        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=key_callback,
        ) as viewer:

            # Disable rangefinder visualization.
            viewer.opt.flags[
                mujoco.mjtVisFlag.mjVIS_RANGEFINDER
            ] = 0

            viewer.sync()

            # Before the loop as well: a renderer that starts later would
            # otherwise hold the spawn pose until the first object packet.
            self.publish_objects(rate=1e9)
            self.publish_state(rate=1e9)

            # The flag behind the viewer's Pause button. Private, so treat its
            # absence as non-fatal: without it the sim simply cannot be paused
            # from the window, which is how it behaved before.
            try:
                ui = viewer._get_sim()
                self._running = bool(ui.run)   # read-only: read it, never assign
                self._ui = ui
                print("[sim] pause: the viewer's Pause button and SPACE control "
                      "this loop", flush=True)
                self._running = bool(ui.run) != self._invert
            except Exception as exc:
                print(f"[sim] pause unavailable from the viewer ({exc}); "
                      f"use kill -USR1 $(pgrep -f run_mujoco_wbc_sim)",
                      flush=True)

            if not self._running:
                print("[sim] started PAUSED - press the viewer's Pause button "
                      "or SPACE to run", flush=True)

            last_print = time.time()

            # Physics rate and render rate are NOT the same rate.
            #
            # This loop used to do exactly one mj_step per viewer.sync(). sync()
            # costs ~13 ms, so the intended sleep(sim_dt - elapsed) never slept
            # and physics advanced at render rate: 65 steps/s x 2 ms = 0.13 s of
            # sim per wall second, i.e. 13% of real time. Nothing looked broken -
            # the sim just ran in slow motion, which is fatal for teleop, because
            # the WBC and the operator both act in WALL time. A 50 Hz controller
            # against a 13%-speed sim moves the robot an eighth as far per
            # command, which reads as "the robot barely responds".
            #
            # So: step until sim time has caught up with wall time, then render
            # at most SYNC_HZ. The step budget per iteration bounds the catch-up
            # so a slow patch cannot spiral.
            SYNC_HZ = 60.0
            MAX_STEPS_PER_ITER = 100

            wall_ref = time.time()
            sim_ref = self.data.time
            last_time = self.data.time
            udp_reset = False
            self._last_visual_reset = 0.0
            last_sync = 0.0
            rtf = 1.0

            while viewer.is_running():

                loop_start = time.time()

                # Backspace in the viewer calls mj_resetData behind our back;
                # the only observable is sim time jumping backwards.
                # A reset request does exactly what backspace does, and then
                # takes the same path below.
                want_reset, place = self.poll_control()
                if want_reset:
                    with viewer.lock():
                        mujoco.mj_resetData(self.model, self.data)
                        mujoco.mj_forward(self.model, self.data)
                    print("[sim] reset (requested over udp)", flush=True)
                    udp_reset = True
                if place:
                    with viewer.lock():
                        self.place_objects(place)
                if self.data.time < last_time - 1e-9:
                    # WHO reset it? Three things make sim time jump back:
                    #   backspace in the viewer, our own reset request, and
                    #   MuJoCo itself - when the physics blows up (NaN / huge
                    #   acceleration) mj_step silently calls mj_resetData and
                    #   leaves a BADQACC/QVEL/QPOS warning behind. That last one looks
                    #   exactly like a second backspace nobody pressed.
                    bad = next((self.data.warning[int(w)] for w in (
                        mujoco.mjtWarning.mjWARN_BADQACC,
                        mujoco.mjtWarning.mjWARN_BADQVEL,
                        mujoco.mjtWarning.mjWARN_BADQPOS)
                        if self.data.warning[int(w)].number > 0), None)
                    if bad is not None:
                        dof = int(bad.lastinfo)
                        jn = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                                int(self.model.dof_jntid[dof]))
                              if 0 <= dof < self.model.nv else "?")
                        cause = (f"MuJoCo AUTO-reset: physics went unstable at "
                                 f"dof {dof} ({jn}) at t={last_time:.2f}s")
                        for w in (mujoco.mjtWarning.mjWARN_BADQACC,
                                  mujoco.mjtWarning.mjWARN_BADQVEL,
                                  mujoco.mjtWarning.mjWARN_BADQPOS):
                            self.data.warning[int(w)].number = 0
                        auto = True
                    else:
                        cause = ("udp request" if udp_reset else
                                 f"viewer (backspace) at t={last_time:.2f}s")
                        auto = False
                    udp_reset = False
                    # One press, one look: an unstable-physics reset, or a key
                    # registering twice within 1.5 s, does not start a new
                    # visual episode - the lighting stays that of the reset the
                    # operator actually asked for.
                    dup = (not auto and time.time() - self._last_visual_reset < 1.5)
                    print(f"[sim] reset: {cause}"
                          f"{' - same look kept' if auto or dup else ''}", flush=True)
                    self.apply_spawn_pose()
                    self.randomize_layout()
                    if not (auto or dup):
                        self.new_visual_episode()
                        self._last_visual_reset = time.time()
                    self.log_reset(cause)
                    wall_ref, sim_ref = time.time(), self.data.time
                last_time = self.data.time

                # --------------------------------------------
                # Simulation
                # --------------------------------------------

                # The UI owns pause when it is reachable: the button writes
                # `run`, and reading it every iteration is what makes the
                # button real instead of decorative.
                if self._ui is not None:
                    ui_run = bool(self._ui.run) != self._invert
                    if ui_run != self._running:
                        self._running = ui_run
                        print(f"[sim] {'running' if ui_run else 'PAUSED'} "
                              f"(viewer)", flush=True)

                if self._running:

                    # How much sim time SHOULD have elapsed by now.
                    target = (time.time() - wall_ref) + sim_ref
                    steps = 0

                    while self.data.time < target and steps < MAX_STEPS_PER_ITER:
                        self.step()
                        steps += 1

                    if steps == 0:
                        # Ahead of schedule: give the wall clock a moment rather
                        # than spinning a core for nothing.
                        time.sleep(0.0005)

                elif self._single_step:

                    self.step()
                    self._single_step = False
                    wall_ref, sim_ref = time.time(), self.data.time

                else:
                    # Paused: hold the reference so resuming does not try to
                    # replay the whole pause in one burst.
                    wall_ref, sim_ref = time.time(), self.data.time
                    time.sleep(0.002)

                # Outside the `running` guard on purpose: paused or dragged in
                # the viewer, the renderers should still follow.
                self.publish_objects()
                self.log_poses()
                self.publish_state()

                # --------------------------------------------
                # Viewer - at SYNC_HZ, not once per physics step
                # --------------------------------------------

                if loop_start - last_sync >= 1.0 / SYNC_HZ:
                    self._match_published_framing(viewer)
                    viewer.sync()
                    last_sync = loop_start

                # --------------------------------------------
                # Keyboard / pause
                # --------------------------------------------

                # MuJoCo passive viewer does not expose the
                # same keyboard callback behavior as the old
                # script. Use viewer UI for pause if required.
                #
                # The simulation runs continuously here.

                # --------------------------------------------
                # Status
                # --------------------------------------------

                now = time.time()

                if now - last_print > 2.0:

                    if self.have_wbc_command:
                        status = "WBC COMMAND RECEIVED"
                    else:
                        status = "WAITING FOR WBC"

                    rtf = (self.data.time - sim_ref) / max(now - wall_ref, 1e-6)

                    print(
                        f"[sim] t={self.data.time:.2f}s rtf={rtf:.2f}x"
                        f"{'' if self._running else ' PAUSED'} | "
                        f"{status}",
                        flush=True,
                    )

                    last_print = now
                    wall_ref, sim_ref = now, self.data.time

                # --------------------------------------------
                # Real-time pacing
                # --------------------------------------------

                # No trailing sleep: pacing is the catch-up loop above. Sleeping
                # a fixed sim_dt here is what capped the whole thing at render
                # rate in the first place.

        print("[MuJoCo] viewer closed")


def main() -> int:

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=(
            argparse.RawDescriptionHelpFormatter
        ),
    )

    parser.add_argument(
        "--scene-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--mjcf",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--no-hold",
        action="store_true",
        help=(
            "Do not hold the initial pose while "
            "waiting for WBC commands."
        ),
    )

    parser.add_argument(
        "--paused",
        action="store_true",
        help="Start paused.",
    )

    parser.add_argument(
        "--no-randomize",
        action="store_true",
        help=(
            "Keep the fixed table layout. By default the apple and the basket "
            "get a fresh position on startup and on every viewer reset "
            "(backspace), inside the robot's half of the table and inside "
            "arm's reach."
        ),
    )

    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--fruits",
        default="",
        help="Comma-separated fruits to spawn on the plate (e.g. mango,grapes); "
             "the rest are parked off the table. Default: all. Plate+basket "
             "scenes (layout_random.json) only.",
    )

    parser.add_argument(
        "--reach-min",
        type=float,
        default=0.18,
        help="Closest an object may spawn to the robot base, metres.",
    )

    parser.add_argument(
        "--reach-max",
        type=float,
        default=0.45,
        help="Furthest an object may spawn from the robot base, metres.",
    )

    parser.add_argument(
        "--apple-reach-min",
        type=float,
        default=0.16,
        help="Closest the apple may spawn to the robot base, metres. "
             "Overrides --reach-min for the apple only.",
    )

    parser.add_argument(
        "--apple-reach-max",
        type=float,
        default=0.40,
        help="Furthest the apple may spawn from the robot base, metres. "
             "Overrides --reach-max for the apple only.",
    )

    parser.add_argument(
        "--publish-state",
        default="",
        help=(
            "Comma-separated UDP ports to stream the "
            "robot's pose to, in the same 51-float format "
            "wbc_state_relay.py uses. With this, the relay "
            "process is unnecessary: this IS the DDS bridge, "
            "so mj_data is already here. Keep the relay only "
            "for a sim this script does not own."
        ),
    )

    parser.add_argument(
        "--publish-objects",
        default="",
        help=(
            "Comma-separated UDP ports to stream scene "
            "object poses to, one per renderer, e.g. "
            "'5601,5602'. These are the same ports "
            "wbc_state_relay.py sends the robot on: the "
            "receiver tells the two kinds apart. Without "
            "this the renderers show the objects frozen "
            "at their spawn pose."
        ),
    )

    parser.add_argument(
        "--visual-dr",
        action="store_true",
        help=(
            "Broadcast a fresh visual-randomization seed on every reset "
            "(backspace) on the --publish-objects ports. The renderers apply "
            "it; see scripts/realsim/visual_randomizer.py."
        ),
    )

    parser.add_argument("--visual-seed", type=int, default=None,
                        help="Seed the sequence of visual seeds, for "
                             "reproducible sessions.")

    parser.add_argument(
        "--control-port",
        type=int,
        default=0,
        help="UDP port to accept reset requests on (b'RSRS'), the "
             "programmatic backspace. 0 = off.",
    )

    args = parser.parse_args()

    if not args.mjcf.exists():

        raise FileNotFoundError(
            f"MJCF not found: {args.mjcf}"
        )

    sim = MujocoWBCSim(
        mjcf_path=args.mjcf,
        scene_dir=args.scene_dir,
        no_hold=args.no_hold,
        no_randomize=args.no_randomize,
        seed=args.seed,
        reach_min=args.reach_min,
        reach_max=args.reach_max,
        apple_reach_min=args.apple_reach_min,
        apple_reach_max=args.apple_reach_max,
        paused=args.paused,
        publish_objects=args.publish_objects,
        publish_state=args.publish_state,
        visual_dr=args.visual_dr,
        visual_seed=args.visual_seed,
        control_port=args.control_port,
        fruits=[f.strip() for f in args.fruits.split(",") if f.strip()],
    )

    sim.run()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())