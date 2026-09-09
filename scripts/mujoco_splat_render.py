#!/usr/bin/env python3
"""Hybrid MuJoCo + 3D Gaussian Splatting rendering.

MuJoCo's renderer only draws triangles, so a Gaussian splat cannot be loaded as
a geom. The way round it (the approach MuGS uses) is not to convert the splat at
all: render the splat and the MuJoCo scene separately from the *same* camera and
composite them with MuJoCo's segmentation buffer as the mask.

    MuJoCo camera  ->  gsplat viewmat/K  ->  splat RGB      (background)
    MuJoCo render  ->  RGB + segmentation ->  foreground + mask
    composite = background*(1-mask) + foreground*mask

RoboSnap's splat lives in the "background" frame while the MuJoCo scene is
gravity-aligned, so the camera is mapped across by transform_gravity_from_background.

    python scripts/mujoco_splat_render.py --scene-dir outputs/video_scene
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_gaussians(ply_path: Path, device):
    """Load a 3DGS PLY (same schema robosnap/rendering/render_layered_scene.py reads)."""
    import torch
    from plyfile import PlyData

    v = PlyData.read(str(ply_path))["vertex"]
    names = {p.name for p in v.properties}
    col = lambda *k: np.column_stack([np.asarray(v[n], dtype=np.float32) for n in k])

    means = col("x", "y", "z")
    scales = np.exp(col("scale_0", "scale_1", "scale_2"))
    quats = col("rot_0", "rot_1", "rot_2", "rot_3")
    quats /= np.clip(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12, None)
    opac = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float32)))
    dc = col("f_dc_0", "f_dc_1", "f_dc_2")

    rest = sorted((n for n in names if n.startswith("f_rest_")),
                  key=lambda n: int(n.rsplit("_", 1)[1]))
    if rest:
        r = col(*rest)
        r = r.reshape(len(r), 3, r.shape[1] // 3)
        sh = np.concatenate([dc[:, :, None], r], axis=2).transpose(0, 2, 1)
    else:
        sh = dc[:, None, :]

    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    return dict(means=t(means), scales=t(scales), quats=t(quats),
                opacities=t(opac), sh=t(sh))


def mujoco_camera_to_gsplat(model, data, cam_name, width, height, world_from_render=None):
    """MuJoCo camera -> (viewmat, K) for gsplat.

    MuJoCo cameras look down -Z with +Y up; gsplat expects OpenCV (+Z forward,
    +Y down), so both Y and Z of the camera basis are negated.
    """
    import mujoco

    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cid < 0:
        raise SystemExit(f"no camera named {cam_name!r}")
    pos = data.cam_xpos[cid].copy()
    R_c = data.cam_xmat[cid].reshape(3, 3).copy()      # columns: cam x,y,z in world

    c2w = np.eye(4)
    c2w[:3, :3] = R_c
    c2w[:3, 3] = pos
    if world_from_render is not None:                  # gravity -> splat frame
        c2w = np.linalg.inv(world_from_render) @ c2w

    flip = np.diag([1.0, -1.0, -1.0])                  # MuJoCo -> OpenCV
    c2w[:3, :3] = c2w[:3, :3] @ flip
    viewmat = np.linalg.inv(c2w).astype(np.float32)

    fovy = float(model.cam_fovy[cid])
    fy = (height / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
    K = np.array([[fy, 0, width / 2.0], [0, fy, height / 2.0], [0, 0, 1]], dtype=np.float32)
    return viewmat, K


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument("--mjcf", type=Path, help="default: <scene>/mujoco/scene_with_robot.xml")
    p.add_argument("--ply", type=Path, help="default: <scene>/gravity_aligned_background.ply")
    p.add_argument("--gravity-json", type=Path, help="default: <scene>/gravity_alignment.json")
    p.add_argument("--camera", default="scene_view")
    p.add_argument("--fovy", type=float, default=58.0)
    p.add_argument("--intrinsics-json", type=Path,
                   help="VGGT camera.json; derives fovy + gsplat K from the real "
                        "camera instead of guessing. Default: <scene>/depth/camera.json")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--steps", type=int, default=0, help="physics steps before rendering")
    p.add_argument("--hold", action="store_true",
                   help="Drive position actuators to their initial joint angles so an "
                        "unactuated humanoid does not collapse under gravity.")
    p.add_argument("--output", type=Path)
    p.add_argument("--capture-camera", action="store_true",
                   help="Render from the original capture viewpoint recorded in "
                        "gravity_alignment.json (transform_gravity_from_camera).")
    p.add_argument("--bg-fill", default="", metavar="R,G,B",
                   help="Colour for pixels the splat does not cover (0-255). "
                        "Without it, uncovered pixels are black.")
    p.add_argument("--exclude-geoms", default="floor",
                   help="Comma-separated geom/body substrings kept as background.")
    p.add_argument("--no-transform", action="store_true",
                   help="splat already in the MuJoCo world frame")
    a = p.parse_args()

    import torch, mujoco
    from gsplat import rasterization
    from PIL import Image

    mjcf = a.mjcf or a.scene_dir / "mujoco" / "scene_with_robot.xml"
    ply = a.ply
    if ply is None:                      # newest gravity-aligned background
        cands = sorted(a.scene_dir.glob("gravity_aligned_background*.ply"),
                       key=lambda q: q.stat().st_mtime, reverse=True)
        if not cands:
            raise SystemExit(f"no gravity_aligned_background*.ply in {a.scene_dir}")
        ply = cands[0]
        print(f"[auto] background: {ply.name}")
    gjson = a.gravity_json or a.scene_dir / "gravity_alignment.json"
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    cam_name = a.camera
    mjcf_to_load = mjcf

    ijson = a.intrinsics_json or (a.scene_dir / "depth" / "camera.json")
    if ijson.is_file():
        cj = json.loads(ijson.read_text(encoding="utf-8"))
        K0 = np.asarray(cj["intrinsic_original_pixels"], dtype=np.float64)
        ow, oh = cj["original_size_wh"]
        # scale intrinsics from the capture resolution to the render resolution
        sx, sy = a.width / float(ow), a.height / float(oh)
        fx, fy = K0[0, 0] * sx, K0[1, 1] * sy
        a.fovy = float(np.rad2deg(2 * np.arctan((a.height / 2.0) / fy)))
        print(f"[hybrid] real intrinsics {ow}x{oh}: fx={K0[0,0]:.1f} fy={K0[1,1]:.1f} "
              f"-> fovy={a.fovy:.2f} deg at {a.width}x{a.height}")
    if a.capture_camera:
        T = np.asarray(json.loads(gjson.read_text())["transform_gravity_from_camera"],
                       dtype=np.float64)                       # c2w, OpenCV convention
        R = T[:3, :3] @ np.diag([1.0, -1.0, -1.0])             # OpenCV -> MuJoCo camera
        M = np.eye(4); M[:3, :3] = R
        import trimesh
        q = trimesh.transformations.quaternion_from_matrix(M)  # (w,x,y,z)
        pos = T[:3, 3]
        cam_xml = (f'<camera name="capture" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" '
                   f'quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}" fovy="{a.fovy}"/>')
        text = mjcf.read_text(encoding="utf-8").replace("</worldbody>", cam_xml + "</worldbody>")
        mjcf_to_load = mjcf.with_name("_capture_" + mjcf.name)
        mjcf_to_load.write_text(text, encoding="utf-8")
        cam_name = "capture"
        print(f"[hybrid] capture camera at {np.round(pos,3)} fovy={a.fovy}")

    model = mujoco.MjModel.from_xml_path(str(mjcf_to_load))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    hold_targets = None
    if a.hold and model.nu:
        hold_targets = np.zeros(model.nu)
        for i in range(model.nu):
            j = int(model.actuator_trnid[i, 0])
            if j >= 0 and int(model.jnt_type[j]) in (mujoco.mjtJoint.mjJNT_HINGE,
                                                     mujoco.mjtJoint.mjJNT_SLIDE):
                hold_targets[i] = data.qpos[int(model.jnt_qposadr[j])]
        print(f"[hybrid] holding {model.nu} actuators at their initial angles")
    for _ in range(a.steps):
        if hold_targets is not None:
            data.ctrl[:] = hold_targets
        mujoco.mj_step(model, data)

    world_from_render = None
    if not a.no_transform and gjson.is_file():
        j = json.loads(gjson.read_text(encoding="utf-8"))
        T = j.get("transform_gravity_from_background")
        if T is not None:
            world_from_render = np.asarray(T, dtype=np.float64)
            print("[hybrid] applying transform_gravity_from_background to the camera")

    g = load_gaussians(ply, dev)
    print(f"[hybrid] {len(g['means']):,} gaussians, sh degree {g['sh'].shape[1]}")

    viewmat, K = mujoco_camera_to_gsplat(model, data, cam_name, a.width, a.height,
                                         world_from_render)
    with torch.no_grad():
        colors, alphas, _ = rasterization(
            means=g["means"], quats=g["quats"], scales=g["scales"],
            opacities=g["opacities"], colors=g["sh"],
            sh_degree=int(np.sqrt(g["sh"].shape[1]) - 1),
            viewmats=torch.from_numpy(viewmat).to(dev)[None],
            Ks=torch.from_numpy(K).to(dev)[None],
            width=a.width, height=a.height, packed=False,
        )
    background = (colors[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    cov = alphas[0].clamp(0, 1).cpu().numpy()
    if cov.ndim == 3:
        cov = cov[..., 0]
    print(f"[hybrid] splat rendered, mean alpha {float(alphas.mean()):.3f} "
          f"({100*(cov > 0.1).mean():.1f}% of pixels covered)")
    if a.bg_fill:
        fill = np.array([float(x) for x in a.bg_fill.split(",")], dtype=np.float32)
        w = np.clip(cov, 0.0, 1.0)[..., None]
        background = (background * w + fill * (1 - w)).astype(np.uint8)
        print(f"[hybrid] uncovered pixels filled with {fill.astype(int).tolist()}")

    renderer = mujoco.Renderer(model, a.height, a.width)
    vopt = mujoco.MjvOption(); mujoco.mjv_defaultOption(vopt)
    vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
    renderer.update_scene(data, camera=cam_name, scene_option=vopt)
    foreground = renderer.render().copy()
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=cam_name, scene_option=vopt)
    seg = renderer.render()[:, :, 0].astype(np.int32)
    renderer.disable_segmentation_rendering()
    renderer.close()

    # Foreground = only the geoms we want drawn over the splat. Everything else
    # (floor plane, skybox) must stay transparent or the background never shows.
    exclude = {e.strip() for e in a.exclude_geoms.split(",") if e.strip()}
    keep_ids = []
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        bid = int(model.geom_bodyid[gid])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if any(x and (x in gname or x in bname) for x in exclude):
            continue
        if bid == 0:                       # worldbody geoms (floor) are background
            continue
        if int(model.geom_group[gid]) == 3:   # collision-only geoms are not visual
            continue
        keep_ids.append(gid)
    mask = np.isin(seg, keep_ids).astype(np.float32)
    print(f"[hybrid] {len(keep_ids)} foreground geoms, covering "
          f"{100*mask.mean():.1f}% of pixels")

    comp = background * (1 - mask[..., None]) + foreground * mask[..., None]
    out = a.output or a.scene_dir / "hybrid_render.png"
    Image.fromarray(comp.astype(np.uint8)).save(out)
    Image.fromarray(background).save(out.with_name(out.stem + "_bg.png"))
    print(f"[hybrid] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
