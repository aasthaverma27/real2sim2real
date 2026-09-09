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
        --scene-dir outputs/g1_ok \
        --mjcf outputs/g1_ok/mujoco/scene_g1.xml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

# The wire format lives next to its reader, in the relay module.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wbc_state_relay import pack_objects  # noqa: E402

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
        paused: bool = False,
        publish_objects: str = "",
    ):
        self.mjcf_path = Path(mjcf_path)
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
        # Optional initial joint hold
        #
        # This is only used until WBC starts sending commands.
        # Once WBC commands arrive, the WBC commands take over.
        # ----------------------------------------------------

        self.initial_hold = None

        if not no_hold:
            self.initial_hold = self._get_initial_joint_positions()

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

        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
        ) as viewer:

            # Disable rangefinder visualization.
            viewer.opt.flags[
                mujoco.mjtVisFlag.mjVIS_RANGEFINDER
            ] = 0

            viewer.sync()

            # Before the loop as well: a renderer that starts later would
            # otherwise hold the spawn pose until the first object packet.
            self.publish_objects(rate=1e9)

            running = not self.paused

            last_print = time.time()

            while viewer.is_running():

                loop_start = time.time()

                # --------------------------------------------
                # Simulation
                # --------------------------------------------

                if running:

                    self.step()

                # Outside the `running` guard on purpose: paused or dragged in
                # the viewer, the renderers should still follow.
                self.publish_objects()

                # --------------------------------------------
                # Viewer
                # --------------------------------------------

                viewer.sync()

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

                    print(
                        f"[sim] t={self.data.time:.2f}s | "
                        f"{status}",
                        flush=True,
                    )

                    last_print = now

                # --------------------------------------------
                # Real-time pacing
                # --------------------------------------------

                elapsed = time.time() - loop_start

                sleep_time = (
                    self.sim_dt - elapsed
                )

                if sleep_time > 0:

                    time.sleep(
                        sleep_time
                    )

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

    args = parser.parse_args()

    if not args.mjcf.exists():

        raise FileNotFoundError(
            f"MJCF not found: {args.mjcf}"
        )

    sim = MujocoWBCSim(
        mjcf_path=args.mjcf,
        scene_dir=args.scene_dir,
        no_hold=args.no_hold,
        paused=args.paused,
        publish_objects=args.publish_objects,
    )

    sim.run()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())