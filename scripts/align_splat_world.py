#!/usr/bin/env python3
"""Align a COLMAP-frame splat to the scene's world coordinates.

The transform is a similarity, and each part comes from a different constraint
because no single one determines all of it:

  gravity   the thin direction of the camera-centre cloud. A handheld orbit is
            a flat ring, so its normal is up. Scale-free and robust.
  scale     the table's radius, 0.418 m, measured off the disc the splat
            reconstructs at tabletop height. Cross-checked against the
            floor-to-tabletop separation and the phone's height; the three
            agree to about 10%.
  z, xy     the tabletop plane goes to world z=0.713 and the disc centre to the
            table's world xy.
  yaw       swept here, because a round table carries no yaw information and
            the one camera shared with the splat's own frame is 0.4 s off. Yaw
            IS observable from a single view (unlike scale, which cancels in the
            projection), so it is found by rendering against the photograph.

The sweep rotates the CAMERA rather than the splat: transforming 3.6M gaussians
per candidate would be far slower, and rendering the untouched splat from an
inversely-transformed camera gives an identical image.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def yaw_mat(deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def basis_from_up(up: np.ndarray) -> np.ndarray:
    """Rotation taking the splat's up axis to +z (no yaw component fixed)."""
    up = up / np.linalg.norm(up)
    ax = np.cross(up, [0.0, 0.0, 1.0])
    s = np.linalg.norm(ax)
    if s < 1e-9:
        return np.eye(3)
    ax /= s
    th = np.arccos(np.clip(up @ np.array([0.0, 0.0, 1.0]), -1.0, 1.0))
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ply", type=Path, default=Path("outputs/g1_ok/gs_wide.ply"))
    p.add_argument("--poses", type=Path,
                   default=Path("outputs/g1_ok/roommesh/colmap_poses.json"))
    p.add_argument("--photo", type=Path, default=Path("outputs/g1_ok/image.png"))
    p.add_argument("--scene", type=Path,
                   default=Path("outputs/g1_ok/mujoco/scene_g1.xml"))
    p.add_argument("--out", type=Path,
                   default=Path("outputs/g1_ok/gs_wide_align.json"))
    p.add_argument("--scale", type=float, default=0.41)
    p.add_argument("--table-h", type=float, default=-1.1550,
                   help="tabletop height along the up axis, splat units")
    p.add_argument("--table-xy", type=float, nargs=2, default=[0.648, 0.676],
                   help="tabletop disc centre in the horizontal basis, splat units")
    p.add_argument("--world-table", type=float, nargs=3, default=[-0.055, 0.724, 0.713])
    p.add_argument("--yaw-step", type=float, default=3.0)
    p.add_argument("--render-long", type=int, default=480)
    p.add_argument("--opacity", type=float, default=0.02)
    p.add_argument("--score-top", type=float, default=1.0,
                   help="score only the top fraction of the frame. The table is "
                        "pinned by construction, so it matches at many yaws and "
                        "swamps the score; the wall behind it is what actually "
                        "carries yaw information. 0.45 scores wall only.")
    a = p.parse_args()

    import cv2
    import mujoco
    import torch
    from gsplat import rasterization
    from plyfile import PlyData

    dev = "cuda"

    # ---- the splat, on the GPU once ------------------------------------------
    v = PlyData.read(str(a.ply))["vertex"]
    keep = (1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32)))) > a.opacity
    means = np.stack([v["x"], v["y"], v["z"]], 1)[keep].astype(np.float32)
    quats = np.stack([v[f"rot_{i}"] for i in range(4)], 1)[keep].astype(np.float32)
    scales = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1)[keep]).astype(np.float32)
    opac = (1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32))))[keep].astype(np.float32)
    dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], 1)[keep].astype(np.float32)
    rgb = np.clip(0.5 + 0.2820948 * dc, 0.0, 1.0)
    print(f"[align] {len(means):,} gaussians kept of {len(keep):,}", flush=True)

    t_means = torch.from_numpy(means).to(dev)
    t_quats = torch.from_numpy(quats).to(dev)
    t_scales = torch.from_numpy(scales).to(dev)
    t_opac = torch.from_numpy(opac).to(dev)
    t_rgb = torch.from_numpy(rgb).to(dev)

    # ---- the world-side camera ----------------------------------------------
    model = mujoco.MjModel.from_xml_path(str(a.scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    ci = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cap")
    Ccap = data.cam_xpos[ci].copy()
    Rocv = data.cam_xmat[ci].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    fovy = float(model.cam_fovy[ci])

    photo = cv2.imread(str(a.photo))
    H = a.render_long
    W = int(round(H * photo.shape[1] / photo.shape[0]))
    photo = cv2.resize(photo, (W, H), interpolation=cv2.INTER_AREA)
    gph = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY).astype(np.float32)
    f = (H / 2) / np.tan(np.deg2rad(fovy) / 2)
    K = torch.tensor([[[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]]],
                     dtype=torch.float32, device=dev)

    # ---- the fixed part of the transform ------------------------------------
    poses = json.load(open(a.poses))
    C = np.array([poses[k]["C"] for k in poses])
    up = np.linalg.svd(C - C.mean(0), full_matrices=False)[2][2]
    if (means[::97] @ up < np.median(C @ up)).mean() < 0.5:
        up = -up
    R0 = basis_from_up(up)
    b1 = np.cross(up, [0.0, 0.0, 1.0]); b1 /= np.linalg.norm(b1)
    b2 = np.cross(up, b1)
    # the table's anchor point in splat coordinates
    anchor = a.table_xy[0] * b1 + a.table_xy[1] * b2 + a.table_h * up
    world_table = np.asarray(a.world_table, float)

    def transform(yaw):
        """world = s * Ryaw @ R0 @ splat + t, with the table pinned."""
        R = yaw_mat(yaw) @ R0
        s = a.scale
        t = world_table - s * (R @ anchor)
        return s, R, t

    def render(yaw):
        s, R, t = transform(yaw)
        # invert onto the camera instead of moving 3.6M gaussians
        Rw2s = R.T / s
        Ccam_s = Rw2s @ (Ccap - t)
        Rcam_s = Rw2s @ Rocv * s          # keep the rotation orthonormal
        Rcam_s = Rcam_s / np.linalg.norm(Rcam_s[:, 0])
        w2c = np.eye(4)
        w2c[:3, :3] = Rcam_s.T
        w2c[:3, 3] = -Rcam_s.T @ Ccam_s
        vm = torch.from_numpy(w2c.astype(np.float32))[None].to(dev)
        with torch.no_grad():
            img, _, _ = rasterization(t_means, t_quats, t_scales, t_opac, t_rgb,
                                      vm, K, W, H, packed=False)
        return (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    band = np.zeros((H, W), bool)
    band[: int(H * a.score_top), :] = True

    def ncc(yaw):
        r = render(yaw)
        g = cv2.cvtColor(r, cv2.COLOR_RGB2GRAY).astype(np.float32)
        m = (g > 3) & band
        if m.sum() < 0.03 * g.size:
            return -1.0, float(m.mean())
        x = g[m] - g[m].mean()
        y = gph[m] - gph[m].mean()
        d = np.sqrt((x * x).sum() * (y * y).sum())
        return (float((x * y).sum() / d) if d > 0 else -1.0), float(m.mean())

    print(f"[align] sweeping yaw at {a.yaw_step} deg, render {W}x{H}", flush=True)
    best = (-9.0, 0.0)
    for yaw in np.arange(0.0, 360.0, a.yaw_step):
        val, cov = ncc(float(yaw))
        star = ""
        if val > best[0]:
            best = (val, float(yaw))
            star = "  *"
        print(f"  yaw {yaw:6.1f}  NCC {val:+.4f}  coverage {100*cov:5.1f}%{star}",
              flush=True)
    # refine
    for step in (1.0, 0.25):
        c = best[1]
        for yaw in np.arange(c - 2 * step, c + 2 * step + 1e-9, step):
            val, _ = ncc(float(yaw))
            if val > best[0]:
                best = (val, float(yaw))
        print(f"[align] refined at {step} deg -> yaw {best[1]:.2f}, NCC {best[0]:+.4f}",
              flush=True)

    s, R, t = transform(best[1])
    print(f"\n[align] BEST yaw {best[1]:.2f} deg, NCC {best[0]:+.4f}")
    print(f"  scale {s:.4f}")
    print(f"  R\n{np.round(R,5)}")
    print(f"  t {np.round(t,4)}")
    a.out.write_text(json.dumps(dict(scale=s, R=R.tolist(), t=t.tolist(),
                                     yaw_deg=best[1], ncc=best[0],
                                     up=up.tolist()), indent=1))
    print(f"[align] wrote {a.out}")
    import cv2 as _c
    _c.imwrite(str(a.out.with_suffix(".png")),
               np.concatenate([photo, render(best[1])[:, :, ::-1]], 1))
    print(f"[align] wrote {a.out.with_suffix('.png')}  (photo | splat at that yaw)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
