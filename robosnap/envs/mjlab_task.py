"""Register a reconstructed RoboSnap scene as an mjlab task.

mjlab's `train` and `play` only accept registered tasks, so a reconstructed scene
has to become one before any of mjlab's machinery - PPO, checkpoints, tensorboard,
`play --viewer` - applies to it. This builds that task: mjlab's Unitree G1 with its
base welded, standing at the reconstructed table, reaching for one of the
reconstructed objects, optionally observing the MuGS hybrid image.

The scene entities come straight out of the RoboSnap export: each object is
lifted from the exported MJCF as its own mjlab entity, so the table is static and
the manipulable objects have free joints, exactly as exported.

    bash scripts/run_train.sh Mjlab-RoboSnap-G1-Reach --env.scene.num-envs 4096
    bash scripts/run_play.sh  Mjlab-RoboSnap-G1-Reach --viewer native
"""
from __future__ import annotations

import copy
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE = ROOT / "outputs" / "video_scene"
TASK_ID = "Mjlab-RoboSnap-G1-Reach"


# ── scene entities from the exported MJCF ────────────────────────────────
def _single_body_xml(scene_xml: Path, body_name: str, out: Path) -> Path:
    """Write an MJCF containing one body of the exported scene, with its assets.

    mjlab wants one entity per manipulable thing, while export_mujoco.py writes a
    single scene. Filtering by body keeps the exported meshes, poses and collision
    hulls exactly as they are rather than rebuilding them.
    """
    tree = ET.parse(scene_xml)
    root = tree.getroot()
    world = root.find("worldbody")
    if world is None:
        raise SystemExit(f"{scene_xml} has no worldbody")

    keep = None
    for b in list(world.findall("body")):
        if b.get("name") == body_name:
            keep = b
        world.remove(b)
    if keep is None:
        raise SystemExit(f"no body named {body_name!r} in {scene_xml}")
    for g in list(world.findall("geom")) + list(world.findall("camera")):
        world.remove(g)          # drop the synthetic floor and scene cameras
    world.append(keep)

    # keep only the assets this body references
    used = set()
    for g in keep.iter("geom"):
        for attr in ("mesh", "material"):
            if g.get(attr):
                used.add(g.get(attr))
    asset = root.find("asset")
    if asset is not None:
        for el in list(asset):
            name = el.get("name")
            if el.tag == "texture" and el.get("type") == "skybox":
                asset.remove(el)
                continue
            if name and name not in used:
                # a material may be referenced by a kept geom's mesh texture
                if el.tag == "material" and name in used:
                    continue
                if el.tag in ("mesh", "material") and name not in used:
                    asset.remove(el)
    # meshdir/texturedir are relative to the XML, and these filtered files live
    # in a subdirectory, so point them at the export's asset folder absolutely.
    comp = root.find("compiler")
    if comp is not None:
        assets = (scene_xml.parent / (comp.get("meshdir") or "assets")).resolve()
        comp.set("meshdir", str(assets))
        comp.set("texturedir", str(
            (scene_xml.parent / (comp.get("texturedir") or "assets")).resolve()))
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out)
    return out


def _spec_fn(xml_path: Path):
    def make():
        import mujoco

        return mujoco.MjSpec.from_file(str(xml_path))

    return make


def _g1_fixed_cfg(pos, quat):
    """mjlab's G1 with the floating base removed.

    A table-top reach does not want a robot that can fall over, and holding a free
    base in place by rewriting qpos injects joint velocities in the hundreds of
    rad/s. Deleting the joint makes the pelvis part of the world.
    """
    import mujoco
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
        get_g1_robot_cfg,
        get_spec,
    )

    cfg = copy.deepcopy(get_g1_robot_cfg())

    def fixed_spec():
        spec = get_spec()
        for j in list(spec.joints):
            if int(j.type) == int(mujoco.mjtJoint.mjJNT_FREE):
                spec.delete(j)          # MjsJoint has no .delete(); the spec does
        # mjlab wraps a fixed-base entity in a mocap body, and without a reset
        # event that mocap stays at the origin - so the pose is baked onto the
        # root body instead, where it survives the wrapping.
        if len(spec.bodies) > 1:
            root = spec.bodies[1]
            root.pos = list(pos)
            root.quat = list(quat)
        return spec

    cfg.spec_fn = fixed_spec
    cfg.init_state = type(cfg.init_state)(pos=tuple(pos), rot=tuple(quat))
    return cfg


# ── reward ───────────────────────────────────────────────────────────────
def reach_object_reward(env, object_name: str, asset_cfg, sigma: float = 0.20):
    """Dense reach reward: 1 at the object, decaying with distance."""
    import torch
    from mjlab.tasks.manipulation import mdp as manipulation_mdp

    delta = manipulation_mdp.ee_to_object_distance(
        env, object_name=object_name, asset_cfg=asset_cfg)
    dist = torch.linalg.norm(delta, dim=-1)
    return torch.exp(-((dist / sigma) ** 2))


# ── the task ─────────────────────────────────────────────────────────────
def robosnap_g1_reach_env_cfg(
    scene_dir: Path = DEFAULT_SCENE,
    *,
    play: bool = False,
    num_envs: int = 64,
    target_body: str = "obj3_blue_octopus_plush_toy",
    support_body: str = "obj0_round_wooden_table_surface_and_base_fram",
    robot_pos=(0.0, 1.55, -0.74),
    robot_quat=(0.0, 0.0, 0.0, 1.0),
    with_camera: bool = False,
    ply: Path | None = None,
    width: int = 160,
    height: int = 120,
):
    from mjlab.entity import EntityCfg
    from mjlab.envs import ManagerBasedRlEnvCfg, mdp
    from mjlab.managers.action_manager import ActionTermCfg
    from mjlab.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.envs.mdp.actions import JointPositionActionCfg
    from mjlab.scene import SceneCfg
    from mjlab.sim import MujocoCfg, SimulationCfg
    from mjlab.tasks.manipulation import mdp as manipulation_mdp
    from mjlab.viewer import ViewerConfig

    scene_xml = scene_dir / "mujoco" / "scene.xml"
    if not scene_xml.is_file():
        raise SystemExit(f"{scene_xml} not found - run the RoboSnap export first")
    tmp = scene_dir / "mujoco" / "mjlab_entities"
    entities = {"robot": _g1_fixed_cfg(robot_pos, robot_quat)}
    for body in (support_body, target_body):
        xml = _single_body_xml(scene_xml, body, tmp / f"{body[:24]}.xml")
        entities[body] = EntityCfg(spec_fn=_spec_fn(xml))

    palm = SceneEntityCfg("robot", site_names=("right_palm",))
    actor = {
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
        "palm_to_object": ObservationTermCfg(
            func=manipulation_mdp.ee_to_object_distance,
            params={"object_name": target_body, "asset_cfg": palm}),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }
    observations = {
        "actor": ObservationGroupCfg(dict(actor), enable_corruption=not play),
        "critic": ObservationGroupCfg(dict(actor), enable_corruption=False),
    }

    sensors = ()
    if with_camera:
        from mugs_mjlab.sensors import GaussianSensorMjlabCfg

        if ply is None:
            cands = sorted((scene_dir / "splats").glob("*distilled*.ply"),
                           key=lambda q: q.stat().st_mtime, reverse=True)
            if not cands:
                raise SystemExit(
                    f"no splats/*distilled*.ply in {scene_dir} - run "
                    f"scripts/run_distill_splat.sh, or pass ply=")
            ply = cands[0]
        sensors = (GaussianSensorMjlabCfg(
            name="gs_rgb", camera_name=None, parent_body="right_wrist_yaw_link",
            pos=(0.05, 0.0, 0.0), quat=(0.5, 0.5, -0.5, -0.5), fovy=58.0,
            width=width, height=height, background_ply_path=Path(ply),
            render_mode="hybrid", robot_geom_names=[],
            enabled_geom_groups=(1, 2)),)
        observations["actor"].terms["camera"] = ObservationTermCfg(
            func=manipulation_mdp.camera_rgb, params={"sensor_name": "gs_rgb"})

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot", actuator_names=(".*",), scale=0.25,
            use_default_offset=True)}

    rewards = {
        "reach": RewardTermCfg(
            func=reach_object_reward, weight=1.0,
            params={"object_name": target_body, "asset_cfg": palm}),
        "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        "joint_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    }
    terminations = {"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)}

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(num_envs=num_envs, env_spacing=3.0, entities=entities,
                       sensors=sensors),
        observations=observations, actions=actions, commands={}, events={},
        rewards=rewards, terminations=terminations, curriculum={},
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY,
                            entity_name="robot", body_name="torso_link",
                            distance=2.2, elevation=-15.0, azimuth=120.0),
        sim=SimulationCfg(nconmax=200, njmax=1200,
                          mujoco=MujocoCfg(timestep=0.005, iterations=10)),
        episode_length_s=8.0, decimation=4,
    )


def register(scene_dir: Path = DEFAULT_SCENE, with_camera: bool = False) -> str:
    from mjlab.tasks.manipulation.config.yam.rl_cfg import (  # noqa
        YamLiftCubePPORunnerCfg,
    )
    from mjlab.tasks.registry import list_tasks, register_mjlab_task

    if TASK_ID in list_tasks():
        return TASK_ID
    register_mjlab_task(
        TASK_ID,
        env_cfg=robosnap_g1_reach_env_cfg(scene_dir, with_camera=with_camera),
        play_env_cfg=robosnap_g1_reach_env_cfg(scene_dir, play=True, num_envs=16,
                                               with_camera=with_camera),
        rl_cfg=YamLiftCubePPORunnerCfg(),
    )
    return TASK_ID
