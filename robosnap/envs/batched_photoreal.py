"""Batched photoreal environment over a reconstructed RoboSnap scene.

Physics runs in MuJoCo Warp, the foreground is warp's batched mesh renderer, and
the background is a Gaussian splat rasterised for every (camera, world) pair in
one gsplat call. Nothing leaves the GPU between physics and observation.

The environment owns the simulation; a Task owns everything scene-specific -
which bodies matter, what the reward is, when an episode is over. Swapping the
task should not require touching this file.

    cfg = EnvConfig(scene_dir=Path("outputs/video_scene"), num_worlds=256)
    env = BatchedPhotorealEnv(cfg, ReachTask("g1_right_wrist_yaw_link",
                                             "obj3_blue_octopus_plush_toy"))
    obs = env.reset()
    obs, reward, done, info = env.step(actions)

Requires the batch-ready MJCF from scripts/warp_gate.py --emit: with the MJCF as
exported, physics is about a thousand times slower (see
docs/batched_photoreal_env.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import logging as _logging

import numpy as np

SH_C0 = 0.28209479177387814


@dataclass
class EnvConfig:
    scene_dir: Path
    mjcf: Path | None = None
    ply: Path | None = None
    num_worlds: int = 64
    cameras: list[str] = field(default_factory=lambda: ["capture"])
    width: int = 240
    height: int = 427
    horizon: int = 100
    """Steps before an episode is truncated."""
    control_decimation: int = 10
    """Physics steps per env step: 10 x 2 ms = 50 Hz control."""
    action_scale: float = 0.25
    """Radians of joint target offset per unit action."""
    actuator_filter: str = "right_shoulder,right_elbow,right_wrist"
    """Substrings selecting the controlled actuators; the rest hold position."""
    randomize_objects: float = 0.12
    randomize_physics: float = 0.0
    pin_base_prefix: str = "g1_"
    bg_fill: tuple[float, float, float] | None = (40.0, 40.0, 45.0)
    render_every_step: bool = True
    """False skips image observations - useful for physics-only throughput."""
    nconmax: int = 128
    njmax: int = 512
    seed: int = 0
    vram_cap_gb: float = 31.0
    device: str = "cuda"


class ReachTask:
    """Move a robot link to an object.

    Reward is the negative distance from the link to the target, plus a bonus on
    success, so it is dense and needs no shaping constants per scene. Success ends
    the episode; the horizon truncates it.
    """

    def __init__(self, link_body: str, target_body: str, success_radius: float = 0.08,
                 success_bonus: float = 1.0):
        self.link_body = link_body
        self.target_body = target_body
        self.success_radius = success_radius
        self.success_bonus = success_bonus

    def bind(self, mjm, mujoco) -> None:
        self.link_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, self.link_body)
        self.target_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY,
                                           self.target_body)
        for name, i in ((self.link_body, self.link_id),
                        (self.target_body, self.target_id)):
            if i < 0:
                raise SystemExit(f"ReachTask: no body named {name!r}")

    def features(self, xpos):
        """(delta vector, distance) per world - also the task's observation."""
        delta = xpos[:, self.target_id] - xpos[:, self.link_id]
        return delta, delta.norm(dim=-1)

    def reward_and_success(self, xpos):
        _, dist = self.features(xpos)
        success = dist < self.success_radius
        return -dist + self.success_bonus * success.float(), success


class BatchedPhotorealEnv:
    def __init__(self, cfg: EnvConfig, task):
        import mujoco
        import mujoco_warp as mw
        import torch
        import warp as wp
        
        # Warp logs every kernel-cache hit at default level, which buries
        # the actual output under 40 lines of module loads.
        wp.config.log_level = _logging.WARNING  # warp 1.13 has no wp.LOG_*

        self._mujoco, self._mw, self._wp, self._torch = mujoco, mw, wp, torch
        self.cfg = cfg
        self.task = task
        self.device = cfg.device

        free, total = torch.cuda.mem_get_info()
        if (total - free) / 2**30 >= cfg.vram_cap_gb:
            raise SystemExit(f"GPU already holds {(total - free) / 2**30:.1f} GiB, "
                             f"at or over the {cfg.vram_cap_gb:.0f} GiB cap")
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, cfg.vram_cap_gb / (total / 2**30)))

        mjcf = cfg.mjcf or cfg.scene_dir / "mujoco" / "scene_batch.xml"
        if not mjcf.is_file():
            raise SystemExit(f"{mjcf} not found - emit it with:\n"
                             f"  bash scripts/run_warp_gate.sh --scene-dir "
                             f"{cfg.scene_dir} --emit {mjcf}")
        self.mjm = mujoco.MjModel.from_xml_path(str(mjcf))
        self.mjd = mujoco.MjData(self.mjm)
        mujoco.mj_forward(self.mjm, self.mjd)
        if task is not None:
            task.bind(self.mjm, mujoco)

        self.cids = []
        for name in cfg.cameras:
            i = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if i < 0:
                raise SystemExit(f"no camera {name!r} in {mjcf}")
            self.cids.append(i)
        self.ncam = len(self.cids)
        self.npix = cfg.width * cfg.height

        self._build_actuators()
        self._build_reset_state()

        self.m = mw.put_model(self.mjm)
        # Size the contact and constraint buffers from what the scene actually
        # produces at rest, with headroom: convex-hull collision on a settled
        # scene can start with hundreds of constraint rows, and put_data raises
        # rather than growing them. The config values act as floors.
        nconmax = max(cfg.nconmax, 4 * int(self.mjd.ncon) + 64)
        njmax = max(cfg.njmax, 4 * int(self.mjd.nefc) + 256)
        self.d = mw.put_data(self.mjm, self.mjd, nworld=cfg.num_worlds,
                             nconmax=nconmax, njmax=njmax)
        self.nconmax, self.njmax = nconmax, njmax
        if cfg.randomize_physics > 0:
            self._randomize_physics(cfg.randomize_physics)

        self.rc = None
        if cfg.render_every_step:
            self._build_renderer()

        self.qpos = wp.to_torch(self.d.qpos)
        self.qvel = wp.to_torch(self.d.qvel)
        self.ctrl = wp.to_torch(self.d.ctrl)
        self.xpos = wp.to_torch(self.d.xpos)
        self.step_count = torch.zeros(cfg.num_worlds, dtype=torch.int32,
                                      device=self.device)
        self.gen = torch.Generator(device=self.device).manual_seed(cfg.seed)

        # Anything constant per step is uploaded once. Building these inside the
        # loop cost a host-to-device copy per physics step and dominated the step
        # time by ~300x.
        dev = self.device
        self._t_qpos0 = torch.from_numpy(self._qpos0).to(dev)
        self._t_qvel0 = torch.from_numpy(self._qvel0).to(dev)
        self._t_hold = torch.from_numpy(self._hold).to(dev)
        self._t_hold_act = torch.from_numpy(self._hold[self.act_ids]).to(dev)
        self._t_lo = torch.from_numpy(np.asarray(self._ctrl_lo)).to(dev)
        self._t_hi = torch.from_numpy(np.asarray(self._ctrl_hi)).to(dev)
        self._t_pins = [(slice(adr, adr + 7), slice(dadr, dadr + 6),
                         torch.from_numpy(self._qpos0[adr:adr + 7]).to(dev))
                        for adr, dadr in self._pinned]
        self._act_index = torch.tensor(self.act_ids, dtype=torch.long, device=dev)

        # The decimation loop is pure warp, so it captures into a single CUDA
        # graph: one launch per env step instead of ten kernel-launch storms.
        self._graph = None
        try:
            self._mw.step(self.m, self.d)
            wp.synchronize()
            with wp.ScopedCapture() as capture:
                for _ in range(cfg.control_decimation):
                    self._mw.step(self.m, self.d)
            self._graph = capture.graph
        except Exception as e:
            print(f"[env] CUDA graph unavailable ({type(e).__name__}: "
                  f"{str(e)[:100]}), falling back to per-step launches")

    # ── construction helpers ────────────────────────────────────────────
    def _build_actuators(self):
        mujoco = self._mujoco
        wanted = [s.strip() for s in self.cfg.actuator_filter.split(",") if s.strip()]
        self.act_ids = []
        for i in range(self.mjm.nu):
            name = mujoco.mj_id2name(self.mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
            if not wanted or any(w in name for w in wanted):
                self.act_ids.append(i)
        self.num_actions = len(self.act_ids)
        if self.mjm.nu and not self.act_ids:
            raise SystemExit(f"no actuator matches {self.cfg.actuator_filter!r}")
        # A scene with no robot has no actuators at all - that is a valid
        # environment, it just has no actions.

        # hold targets: every actuator sits at its initial joint angle, and the
        # controlled ones move relative to that
        hold = np.zeros(self.mjm.nu, dtype=np.float32)
        for i in range(self.mjm.nu):
            j = int(self.mjm.actuator_trnid[i, 0])
            if j >= 0 and int(self.mjm.jnt_type[j]) in (mujoco.mjtJoint.mjJNT_HINGE,
                                                        mujoco.mjtJoint.mjJNT_SLIDE):
                hold[i] = self.mjd.qpos[int(self.mjm.jnt_qposadr[j])]
        self._hold = hold
        rng = self.mjm.actuator_ctrlrange[self.act_ids]
        self._ctrl_lo = rng[:, 0].astype(np.float32)
        self._ctrl_hi = rng[:, 1].astype(np.float32)
        limited = self.mjm.actuator_ctrllimited[self.act_ids].astype(bool)
        self._ctrl_lo[~limited] = -np.inf
        self._ctrl_hi[~limited] = np.inf

    def _build_reset_state(self):
        mujoco = self._mujoco
        self._qpos0 = self.mjd.qpos.copy().astype(np.float32)
        self._qvel0 = self.mjd.qvel.copy().astype(np.float32)
        self._object_qadr, self._pinned = [], []
        for j in range(self.mjm.njnt):
            if int(self.mjm.jnt_type[j]) != mujoco.mjtJoint.mjJNT_FREE:
                continue
            body = mujoco.mj_id2name(self.mjm, mujoco.mjtObj.mjOBJ_BODY,
                                     int(self.mjm.jnt_bodyid[j])) or ""
            adr = int(self.mjm.jnt_qposadr[j])
            if body.startswith(self.cfg.pin_base_prefix):
                self._pinned.append((adr, int(self.mjm.jnt_dofadr[j])))
            else:
                self._object_qadr.append(adr)

    def _build_renderer(self):
        mw = self._mw
        groups = sorted({int(self.mjm.geom_group[g]) for g in range(self.mjm.ngeom)
                         if int(self.mjm.geom_contype[g]) == 0
                         and int(self.mjm.geom_conaffinity[g]) == 0})
        self.rc = mw.create_render_context(
            self.mjm, nworld=self.cfg.num_worlds,
            cam_res=(self.cfg.width, self.cfg.height),
            render_rgb=True, render_depth=False, render_seg=True,
            enabled_geom_groups=groups,
            cam_active=[i in self.cids for i in range(self.mjm.ncam)])
        active = list(self.rc.cam_id_map.numpy())
        self._adrs = [int(self.rc.rgb_adr.numpy()[active.index(c)]) for c in self.cids]
        self._load_gaussians()

    def _load_gaussians(self):
        import torch
        from plyfile import PlyData

        ply = self.cfg.ply
        if ply is None:
            cands = sorted(self.cfg.scene_dir.glob("splats/*distilled*.ply"),
                           key=lambda q: q.stat().st_mtime, reverse=True)
            if not cands:
                raise SystemExit(
                    f"no splats/*distilled*.ply in {self.cfg.scene_dir} - run "
                    f"scripts/run_distill_splat.sh, or pass ply=")
            ply = cands[0]
        self.ply = ply
        v = PlyData.read(str(ply))["vertex"]
        col = lambda *k: np.column_stack(
            [np.asarray(v[n], dtype=np.float32) for n in k])
        q = col("rot_0", "rot_1", "rot_2", "rot_3")
        q /= np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().to(self.device)
        self.g = dict(
            means=t(col("x", "y", "z")), quats=t(q),
            scales=t(np.exp(col("scale_0", "scale_1", "scale_2"))),
            opacities=t(1.0 / (1.0 + np.exp(
                -np.asarray(v["opacity"], dtype=np.float32)))),
            colors=t(np.clip(0.5 + SH_C0 * col("f_dc_0", "f_dc_1", "f_dc_2"), 0, 1)))

        Ks = []
        for c in self.cids:
            fy = (self.cfg.height / 2) / np.tan(
                np.deg2rad(float(self.mjm.cam_fovy[c])) / 2)
            Ks.append([[fy, 0, self.cfg.width / 2],
                       [0, fy, self.cfg.height / 2], [0, 0, 1]])
        self._K = (torch.tensor(Ks, dtype=torch.float32, device=self.device)
                   .repeat_interleave(self.cfg.num_worlds, dim=0))
        self._flip = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=self.device))
        self._fill = (None if self.cfg.bg_fill is None else
                      torch.tensor(self.cfg.bg_fill, dtype=torch.float32,
                                   device=self.device))

    def _randomize_physics(self, spread: float):
        mujoco, wp = self._mujoco, self._wp
        rng = np.random.default_rng(self.cfg.seed)
        n = self.cfg.num_worlds
        bodies = [int(self.mjm.jnt_bodyid[j]) for j in range(self.mjm.njnt)
                  if int(self.mjm.jnt_type[j]) == mujoco.mjtJoint.mjJNT_FREE
                  and not (mujoco.mj_id2name(self.mjm, mujoco.mjtObj.mjOBJ_BODY,
                                             int(self.mjm.jnt_bodyid[j])) or ""
                           ).startswith(self.cfg.pin_base_prefix)]
        if bodies:
            mass = np.tile(self.mjm.body_mass.astype(np.float32)[None], (n, 1))
            inertia = np.tile(self.mjm.body_inertia.astype(np.float32)[None], (n, 1, 1))
            f = rng.uniform(1 - spread, 1 + spread,
                            size=(n, len(bodies))).astype(np.float32)
            mass[:, bodies] *= f
            inertia[:, bodies, :] *= f[:, :, None]
            self.m.body_mass = wp.array(mass, dtype=wp.float32, ndim=2)
            self.m.body_inertia = wp.array(inertia, dtype=wp.vec3, ndim=2)
        fr = np.tile(self.mjm.geom_friction.astype(np.float32)[None], (n, 1, 1))
        fr[:, :, 0] *= rng.uniform(1 / (1 + spread), 1 + spread,
                                   size=(n, 1)).astype(np.float32)
        self.m.geom_friction = wp.array(fr, dtype=wp.vec3, ndim=2)

    # ── env API ─────────────────────────────────────────────────────────
    def reset(self, world_ids=None):
        """Reset all worlds, or just the ones named. Returns observations."""
        import torch

        ids = (torch.arange(self.cfg.num_worlds, device=self.device)
               if world_ids is None else world_ids)
        if ids.numel() == 0:
            return self.observe()
        self.qpos[ids] = self._t_qpos0
        self.qvel[ids] = self._t_qvel0
        if self.mjm.nu:
            self.ctrl[ids] = self._t_hold

        r = self.cfg.randomize_objects
        if r > 0:
            for adr in self._object_qadr:
                off = (torch.rand((ids.numel(), 2), generator=self.gen,
                                  device=self.device) * 2 - 1) * r
                self.qpos[ids, adr:adr + 2] += off
                yaw = torch.rand((ids.numel(),), generator=self.gen,
                                 device=self.device) * (2 * np.pi)
                self.qpos[ids, adr + 3] = torch.cos(yaw / 2)
                self.qpos[ids, adr + 4] = 0.0
                self.qpos[ids, adr + 5] = 0.0
                self.qpos[ids, adr + 6] = torch.sin(yaw / 2)
        self.step_count[ids] = 0
        self._mw.fwd_position(self.m, self.d)
        return self.observe()

    def step(self, actions=None):
        """actions: (num_worlds, num_actions) in roughly [-1, 1], or None for a
        scene with no actuators (objects and background only)."""
        import torch

        if self.num_actions and actions is not None:
            target = (self._t_hold_act
                      + actions.clamp(-1.0, 1.0) * self.cfg.action_scale)
            self.ctrl[:, self._act_index] = target.clamp(self._t_lo, self._t_hi)

        if self._graph is not None:
            self._wp.capture_launch(self._graph)
        else:
            for _ in range(self.cfg.control_decimation):
                self._mw.step(self.m, self.d)
        # Only needed when the MJCF still has a floating base. The batch MJCF
        # welds it instead (scripts/warp_gate.py --emit), which is both stabler
        # and free; teleporting a base back into place every control step injects
        # joint velocities in the hundreds of rad/s.
        for qs, vs, q0 in self._t_pins:
            self.qpos[:, qs] = q0
            self.qvel[:, vs] = 0.0
        self.step_count += 1

        truncated = self.step_count >= self.cfg.horizon
        if self.task is None:
            reward = torch.zeros(self.cfg.num_worlds, device=self.device)
            success = torch.zeros(self.cfg.num_worlds, dtype=torch.bool,
                                  device=self.device)
        else:
            reward, success = self.task.reward_and_success(self.xpos)
        done = success | truncated
        obs = self.observe()
        info = {"success": success, "truncated": truncated}
        if self.task is not None:
            info["distance"] = self.task.features(self.xpos)[1]
        if done.any():
            self.reset(done.nonzero(as_tuple=True)[0])
        return obs, reward, done, info

    def observe(self) -> dict:
        import torch

        obs = {"qpos": self.qpos.clone(), "xpos": self.xpos.clone()}
        if self.task is not None:
            delta, dist = self.task.features(self.xpos)
            obs["task_delta"], obs["task_distance"] = delta, dist
        if self.rc is not None:
            obs["rgb"] = self.render()
        return obs

    def render(self, components: bool = False):
        """(ncam, num_worlds, height, width, 3) uint8.

        With components=True, also returns the two layers and the mask, which is
        what a viewer needs to show "background only" / "foreground only".
        """
        import torch

        mw, wp = self._mw, self._wp
        mw.render(self.m, self.d, self.rc)
        rgb32 = wp.to_torch(self.rc.rgb_data.view(wp.int32))
        seg32 = wp.to_torch(self.rc.seg_data)
        packed = torch.stack([rgb32[:, s:s + self.npix] for s in self._adrs])
        seg = torch.stack([seg32[:, s:s + self.npix] for s in self._adrs])
        fg = torch.stack([(packed >> 16) & 0xFF, (packed >> 8) & 0xFF,
                          packed & 0xFF], dim=-1).float()
        mask = (seg[..., 1] == int(mw.ObjType.GEOM)).unsqueeze(-1)
        bg = self._background().reshape(self.ncam, self.cfg.num_worlds, self.npix, 3)
        out = torch.where(mask, fg, bg).to(torch.uint8)
        shape = (self.ncam, self.cfg.num_worlds, self.cfg.height, self.cfg.width, 3)
        if not components:
            return out.reshape(shape)
        return {
            "rgb": out.reshape(shape),
            "background": bg.to(torch.uint8).reshape(shape),
            "foreground": fg.to(torch.uint8).reshape(shape),
            "mask": mask.reshape(shape[:-1] + (1,)),
        }

    def _background(self):
        import torch
        from gsplat import rasterization

        wp = self._wp
        n, c = self.cfg.num_worlds, self.ncam
        xpos = wp.to_torch(self.d.cam_xpos)[:, self.cids].transpose(0, 1).reshape(-1, 3)
        xmat = (wp.to_torch(self.d.cam_xmat)[:, self.cids]
                .transpose(0, 1).reshape(-1, 3, 3))
        R = (xmat @ self._flip).transpose(-1, -2)
        nview = c * n
        viewmats = torch.eye(4, device=self.device).expand(nview, 4, 4).contiguous()
        viewmats[:, :3, :3] = R
        viewmats[:, :3, 3] = -(R @ xpos.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            out, alpha, _ = rasterization(
                means=self.g["means"], quats=self.g["quats"], scales=self.g["scales"],
                opacities=self.g["opacities"], colors=self.g["colors"],
                viewmats=viewmats, Ks=self._K, width=self.cfg.width,
                height=self.cfg.height, packed=False)
        rgb = out[..., :3].clamp(0, 1).reshape(nview, self.npix, 3) * 255.0
        if self._fill is not None:
            w = alpha.clamp(0, 1).reshape(nview, self.npix, 1)
            rgb = rgb * w + self._fill * (1 - w)
        return rgb

    def close(self):
        pass
