#!/usr/bin/env python3
"""Render a RoboSnap scene through MuGS's GaussianSensor.

MuJoCo draws the physics, gsplat draws the room, the two are composited - but the renderer is the vendored
third_party/MuGS sensor, so anything built on GaussianSensor (rollout loops,
mjlab batching) works on our scenes.

Three things have to be bridged, because MuGS assumes a COLMAP scene and we
have a reconstruction:

  frame       gravity_aligned_background*.ply is already in the MuJoCo world
              frame (gravity_align_scene bakes transform_gravity_from_background
              into the PLY), so no world_T_gs is needed - unlike MuGS's own
              scenes, which sit in an arbitrary COLMAP frame.
  intrinsics  MuGS derives fx/fy from the MJCF camera's fovy. Our capture has
              real calibration in depth/camera.json, so we pass it explicitly
              and the MJCF fovy is only a fallback.
  masking     MuGS wants a list of robot geom names. Our MJCF names geoms after
              whatever the VLM called the object, so the foreground is selected
              structurally instead: every visual geom that is not the floor,
              not a worldbody geom, and not collision-only.

    bash scripts/run_mugs_render.sh --scene-dir outputs/myroom
    bash scripts/run_mugs_render.sh --scene-dir outputs/myroom \
        --capture-camera --compare outputs/myroom/hybrid_capture.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "MuGS" / "src"))


def apply_vram_cap(cap_gb: float) -> None:
    """Refuse to use more than cap_gb of VRAM.

    2.5M gaussians plus gsplat's per-tile intermediates scale with resolution, so
    a big render can climb fast. set_per_process_memory_fraction makes the
    allocator raise OOM at the cap instead of taking the whole card from whatever
    else is training on it.
    """
    import torch

    if not torch.cuda.is_available() or cap_gb <= 0:
        return
    free, total = torch.cuda.mem_get_info()
    total_gb, used_gb = total / 2**30, (total - free) / 2**30
    if used_gb >= cap_gb:
        raise SystemExit(f"[mugs] GPU already holds {used_gb:.1f} GiB, at or over the "
                         f"{cap_gb:.0f} GiB cap - not starting")
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap_gb / total_gb))
    print(f"[mugs] VRAM cap {cap_gb:.0f} GiB of {total_gb:.1f} GiB "
          f"({used_gb:.1f} GiB already in use)")


def newest_background(scene_dir: Path) -> Path:
    cands = sorted(scene_dir.glob("gravity_aligned_background*.ply"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise SystemExit(f"no gravity_aligned_background*.ply in {scene_dir}")
    return cands[0]


def real_intrinsics(scene_dir: Path, width: int, height: int):
    """(fx, fy, fovy_deg) from the capture's calibration, or None."""
    path = scene_dir / "depth" / "camera.json"
    if not path.is_file():
        return None
    cj = json.loads(path.read_text(encoding="utf-8"))
    K0 = np.asarray(cj["intrinsic_original_pixels"], dtype=np.float64)
    ow, oh = cj["original_size_wh"]
    fx = K0[0, 0] * (width / float(ow))
    fy = K0[1, 1] * (height / float(oh))
    fovy = float(np.rad2deg(2 * np.arctan((height / 2.0) / fy)))
    return fx, fy, fovy


def inject_capture_camera(mjcf: Path, gravity_json: Path, fovy: float) -> tuple[Path, str]:
    """Add a camera at the original capture viewpoint and return the new MJCF."""
    import trimesh

    T = np.asarray(json.loads(gravity_json.read_text())["transform_gravity_from_camera"],
                   dtype=np.float64)                        # c2w, OpenCV convention
    M = np.eye(4)
    M[:3, :3] = T[:3, :3] @ np.diag([1.0, -1.0, -1.0])      # OpenCV -> MuJoCo camera
    q = trimesh.transformations.quaternion_from_matrix(M)   # (w, x, y, z)
    p = T[:3, 3]
    cam = (f'<camera name="capture" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" '
           f'quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}" fovy="{fovy}"/>')
    out = mjcf.with_name("_mugs_" + mjcf.name)
    out.write_text(mjcf.read_text(encoding="utf-8").replace("</worldbody>", cam + "</worldbody>"),
                   encoding="utf-8")
    return out, "capture"


def hide_excluded_objects(model, exclude: set[str]) -> int:
    """Make excluded movable objects fully transparent; returns how many geoms.

    Leaving a geom out of the foreground only stops it being composited: MuJoCo
    still draws it into the shadow map, so an object left out of a run (a fruit
    not on the plate) darkens whatever it sits on. A geom with alpha 0 casts no
    shadow. Only bodies with a free joint are touched; static exclusions (the
    floor, a table the splat shows) keep their look.
    """
    import mujoco

    free = {int(model.jnt_bodyid[j]) for j in range(model.njnt)
            if int(model.jnt_type[j]) == mujoco.mjtJoint.mjJNT_FREE}
    n = 0
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        if bid not in free:
            continue
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if any(x and (x in gname or x in bname) for x in exclude):
            model.geom_rgba[gid, 3] = 0.0
            n += 1
    return n


def foreground_geom_ids(model, exclude: set[str]) -> list[int]:
    """Visual geoms that belong on top of the splat.

    Anything else - the MJCF's checkered floor, the skybox, the collision hulls
    in group 3 - has to stay transparent or the background never shows through.
    """
    import mujoco

    ids = []
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        bid = int(model.geom_bodyid[gid])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if any(x and (x in gname or x in bname) for x in exclude):
            continue
        if bid == 0:                            # worldbody: floor, static decor
            continue
        if int(model.geom_group[gid]) == 3:     # collision-only
            continue
        ids.append(gid)
    return ids


def base_pin(model, data, prefix: str):
    """(qpos slice, qvel slice, initial qpos) for each pinned free joint."""
    import mujoco

    pins = []
    if not prefix:
        return pins
    for j in range(model.njnt):
        if int(model.jnt_type[j]) != mujoco.mjtJoint.mjJNT_FREE:
            continue
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 int(model.jnt_bodyid[j])) or ""
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if not (body.startswith(prefix) or name.startswith(prefix)):
            continue
        qa, va = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
        pins.append((slice(qa, qa + 7), slice(va, va + 6), data.qpos[qa:qa + 7].copy()))
        print(f"[mugs] pinning free base '{name}'")
    return pins


def render_rollout(sensor, model, data, cam_name, camera_params, hold, a) -> int:
    """Step the sim and render every frame through the sensor.

    This is the case the single-frame path cannot check: the GL context and the
    loaded splat have to survive thousands of renders, and with a fixed camera
    the background is rendered once and reused while only the MuJoCo layer is
    redrawn per frame.
    """
    import time

    import cv2
    import mujoco

    pins = base_pin(model, data, a.pin_base)
    steps = max(1, int(round(1.0 / (a.fps * model.opt.timestep))))
    writer = cv2.VideoWriter(str(a.video), cv2.VideoWriter_fourcc(*"mp4v"),
                             a.fps, (a.width, a.height))
    if not writer.isOpened():
        raise SystemExit(f"cannot open {a.video} for writing")
    print(f"[mugs] rollout: {a.frames} frames, {steps} physics steps each "
          f"({model.opt.timestep * 1000:.1f} ms timestep)")
    t0 = time.perf_counter()
    try:
        for i in range(a.frames):
            for _ in range(steps):
                if hold is not None:
                    data.ctrl[:] = hold
                mujoco.mj_step(model, data)
                for qs, vs, q0 in pins:
                    data.qpos[qs] = q0
                    data.qvel[vs] = 0.0
            rgb = sensor.render(model, data, cam_name, camera_params=camera_params)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{a.frames} frames, "
                      f"{(i + 1) / (time.perf_counter() - t0):.1f} render FPS")
    finally:
        writer.release()
    dt = time.perf_counter() - t0
    print(f"[mugs] wrote {a.video} - {a.frames} frames in {dt:.1f}s "
          f"({a.frames / dt:.1f} FPS at {a.width}x{a.height})")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument("--mjcf", type=Path, help="default: <scene>/mujoco/scene_with_robot.xml")
    p.add_argument("--ply", type=Path, help="default: newest gravity_aligned_background*.ply")
    p.add_argument("--gravity-json", type=Path, help="default: <scene>/gravity_alignment.json")
    p.add_argument("--fovy", type=float, default=None,
                   help="Override the vertical FOV used for BOTH layers. Set it "
                        "to the background splat's own fovy when the splat came "
                        "from a different capture than the scene image.")
    p.add_argument("--camera", default="scene_view")
    p.add_argument("--capture-camera", action="store_true",
                   help="render from the original capture viewpoint")
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=1280)
    p.add_argument("--mode", default="hybrid",
                   choices=["hybrid", "3dgs_only", "mujoco_only"])
    p.add_argument("--steps", type=int, default=0, help="physics steps before rendering")
    p.add_argument("--hold", action="store_true",
                   help="drive position actuators to their initial angles so an "
                        "unactuated humanoid does not collapse")
    p.add_argument("--pin-base", default="g1_", metavar="PREFIX",
                   help="hold the floating base of bodies matching this prefix in "
                        "place during --video ('' disables). --hold keeps the "
                        "joints from folding but nothing holds a free base up, so "
                        "without this the robot simply falls out of frame.")
    p.add_argument("--no-depth-composite", action="store_true",
                   help="use MuGS's plain binary mask instead of the depth test")
    p.add_argument("--no-sh", action="store_true",
                   help="DC colour only, as upstream MuGS does")
    p.add_argument("--exclude-geoms", default="floor",
                   help="comma-separated geom/body substrings kept as background")
    p.add_argument("--splat-geoms", default="",
                   help="comma-separated substrings; these geoms keep the mesh's "
                        "silhouette but are painted with the splat's colour, the "
                        "way Re3Sim substitutes its table and ground pixels")
    p.add_argument("--output", type=Path, help="default: <scene>/mugs_render.png")
    p.add_argument("--compare", type=Path,
                   help="report the pixel difference against an existing render")
    p.add_argument("--fps-probe", type=int, default=0,
                   help="re-render N times and report the achieved rate")
    p.add_argument("--video", type=Path, metavar="MP4",
                   help="step physics and render a rollout instead of one frame")
    p.add_argument("--frames", type=int, default=120, help="frames for --video")
    p.add_argument("--fps", type=int, default=30, help="frame rate for --video")
    p.add_argument("--vram-cap-gb", type=float, default=31.0,
                   help="hard VRAM ceiling for this process (0 disables)")
    a = p.parse_args()

    import mujoco
    from PIL import Image
    from mugs.sensors import GaussianSensor, GaussianSensorConfig

    mjcf = a.mjcf or a.scene_dir / "mujoco" / "scene_with_robot.xml"
    ply = a.ply or newest_background(a.scene_dir)
    gjson = a.gravity_json or a.scene_dir / "gravity_alignment.json"

    apply_vram_cap(a.vram_cap_gb)

    intr = real_intrinsics(a.scene_dir, a.width, a.height)
    # The calibration in depth/camera.json belongs to the SCENE's capture image.
    # When the background splat was reconstructed from a different video, its
    # cameras have their own intrinsics, and rendering the splat at the still's
    # wider fovy asks for a field of view the reconstruction never observed -
    # which shows up as black margins that look like a bad splat rather than a
    # mismatched lens.
    fovy = a.fovy if a.fovy else (intr[2] if intr else 58.0)
    if a.fovy:
        # An explicit fovy must drive fx/fy too. Overriding only the reported
        # value left the splat rendered with the still's focal lengths - which
        # here differ by a 1.33 aspect ratio from the background video's own
        # camera, so the splat was sampled over a field of view it never
        # observed and the margins came back black.
        f_px = (a.height / 2.0) / np.tan(np.deg2rad(a.fovy) / 2.0)
        intr = (f_px, f_px, a.fovy)
    if intr:
        print(f"[mugs] real intrinsics -> fx={intr[0]:.1f} fy={intr[1]:.1f} "
              f"(fovy {fovy:.2f} deg at {a.width}x{a.height})")

    cam_name = a.camera
    if a.capture_camera:
        mjcf, cam_name = inject_capture_camera(mjcf, gjson, fovy)
        print(f"[mugs] rendering from the capture viewpoint ({mjcf.name})")

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    hold = None
    if a.hold and model.nu:
        hold = np.zeros(model.nu)
        for i in range(model.nu):
            j = int(model.actuator_trnid[i, 0])
            if j >= 0 and int(model.jnt_type[j]) in (mujoco.mjtJoint.mjJNT_HINGE,
                                                     mujoco.mjtJoint.mjJNT_SLIDE):
                hold[i] = data.qpos[int(model.jnt_qposadr[j])]
    for _ in range(a.steps):
        if hold is not None:
            data.ctrl[:] = hold
        mujoco.mj_step(model, data)
    if a.steps:
        print(f"[mugs] stepped physics {a.steps} steps")

    exclude = {e.strip() for e in a.exclude_geoms.split(",") if e.strip()}
    fg_ids = foreground_geom_ids(model, exclude)

    import mujoco as _mj
    splat_pat = {x.strip() for x in a.splat_geoms.split(",") if x.strip()}
    splat_ids = [g for g in fg_ids
                 if any(x in (_mj.mj_id2name(model, _mj.mjtObj.mjOBJ_GEOM, g) or "")
                        or x in (_mj.mj_id2name(model, _mj.mjtObj.mjOBJ_BODY,
                                 int(model.geom_bodyid[g])) or "")
                        for x in splat_pat)] if splat_pat else None
    if splat_ids:
        print(f"[mugs] {len(splat_ids)} geoms painted from the splat (Re3Sim rule)")

    cfg = GaussianSensorConfig(
        width=a.width, height=a.height,
        background_ply_path=ply,
        render_mode=a.mode,
        robot_geom_ids=fg_ids,
        splat_geom_ids=splat_ids,
        use_sh=not a.no_sh,
        depth_composite=not a.no_depth_composite,
        camera_convention="mujoco",
    )
    sensor = GaussianSensor(cfg)
    print(f"[mugs] {len(fg_ids)} foreground geoms of {model.ngeom}")

    import torch
    peak = lambda: torch.cuda.max_memory_allocated() / 2**30

    camera_params = None
    if intr:
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cid < 0:
            raise SystemExit(f"no camera named {cam_name!r} in {mjcf}")
        camera_params = {
            "position": data.cam_xpos[cid].copy(),
            "rotation": data.cam_xmat[cid].reshape(3, 3).copy(),
            "fx": intr[0], "fy": intr[1],
            "width": a.width, "height": a.height,
        }

    if a.video:
        rc = render_rollout(sensor, model, data, cam_name, camera_params, hold, a)
        sensor.close()
        return rc

    out = sensor.render(model, data, cam_name, return_components=True,
                        camera_params=camera_params)
    rgb = out["rgb"] if isinstance(out, dict) else out

    dest = a.output or a.scene_dir / "mugs_render.png"
    Image.fromarray(rgb).save(dest)
    print(f"[mugs] wrote {dest}")
    if isinstance(out, dict):
        Image.fromarray(out["background"]).save(dest.with_name(dest.stem + "_bg.png"))
        m, v = out["mask"].astype(bool), out["visible"].astype(bool)
        print(f"[mugs] mask {100 * m.mean():.1f}% of pixels, "
              f"visible after depth test {100 * v.mean():.1f}%")
        bd, fd = out.get("background_depth"), out.get("foreground_depth")
        if bd is not None and fd is not None:
            # How far behind each foreground pixel does the splat sit? A gap near
            # zero over a large area means the splat still contains the surface
            # the mesh is drawing - e.g. a background that kept the table.
            sel = m & (bd > 0)
            if sel.any():
                gap = (bd - fd)[sel]
                q = np.percentile(gap, [5, 50, 95])
                print(f"[mugs] splat-behind-mesh gap: p5 {q[0]:+.3f} med {q[1]:+.3f} "
                      f"p95 {q[2]:+.3f} m | within 3 cm: "
                      f"{100 * (np.abs(gap) < 0.03).mean():.1f}% of the mask, "
                      f"within 10 cm: {100 * (np.abs(gap) < 0.10).mean():.1f}%")

    if a.compare and a.compare.is_file():
        ref = np.asarray(Image.open(a.compare).convert("RGB"), dtype=np.int16)
        if ref.shape != rgb.shape:
            print(f"[compare] shape differs: {ref.shape} vs {rgb.shape}")
        else:
            d = np.abs(ref - rgb.astype(np.int16))
            print(f"[compare] vs {a.compare.name}: mean |diff| {d.mean():.2f}, "
                  f"p99 {np.percentile(d, 99):.0f}, max {d.max()}, "
                  f"pixels >16: {100 * (d.max(axis=2) > 16).mean():.2f}%")

    if a.fps_probe:
        import time
        sensor.clear_cache()
        t0 = time.perf_counter()
        for _ in range(a.fps_probe):
            sensor.render(model, data, cam_name, camera_params=camera_params)
        dt = (time.perf_counter() - t0) / a.fps_probe
        print(f"[mugs] {a.fps_probe} renders: {1000 * dt:.2f} ms each ({1 / dt:.1f} FPS) "
              f"at {a.width}x{a.height}")

    print(f"[mugs] peak VRAM {peak():.2f} GiB")
    sensor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
