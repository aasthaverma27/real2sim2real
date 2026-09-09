#!/usr/bin/env python3
"""Fit a splat's scale and yaw against MuJoCo's depth buffer, not its image.

Image appearance cannot determine scale: scaling a scene about the camera
centre leaves every pixel where it was, so the projection cancels `s` exactly
(F12). Fitting yaw from the apple's image position therefore gave a
correct-looking picture with the room 37 cm out of depth, because the apple
constrains the *direction* to itself and not the distance along that ray (F13).

Depth does not cancel. Rendering the splat from an inversely-transformed camera
returns depth in splat units, so world depth is `s * depth`, and `s` is
directly observable.

The objective uses only surfaces that BOTH representations describe - the
table, the apple and the basket, which are real objects the splat reconstructed
and the scene also carries as meshes. There the two depths should be equal. The
robot is excluded: it exists only as a mesh, so the splat behind it is the room
and no agreement is expected.

    scripts/fit_splat_depth.py --ply outputs/g1_ok/gs_wide.ply
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
    up = up / np.linalg.norm(up)
    ax = np.cross(up, [0.0, 0.0, 1.0])
    n = np.linalg.norm(ax)
    if n < 1e-9:
        return np.eye(3)
    ax /= n
    th = np.arccos(np.clip(up @ np.array([0.0, 0.0, 1.0]), -1.0, 1.0))
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ply", type=Path, default=Path("outputs/g1_ok/gs_wide.ply"))
    p.add_argument("--poses", type=Path,
                   default=Path("outputs/g1_ok/roommesh/colmap_poses.json"))
    p.add_argument("--scene", type=Path,
                   default=Path("outputs/g1_ok/mujoco/scene_g1.xml"))
    p.add_argument("--out", type=Path, default=Path("outputs/g1_ok/gs_depth_fit.json"))
    p.add_argument("--table-h", type=float, default=-1.1550)
    p.add_argument("--table-xy", type=float, nargs=2, default=[0.648, 0.676])
    p.add_argument("--world-table", type=float, nargs=3, default=[-0.055, 0.724, 0.713])
    p.add_argument("--scale-range", type=float, nargs=3, default=[0.30, 0.60, 0.02],
                   metavar=("LO", "HI", "STEP"))
    p.add_argument("--yaw-step", type=float, default=10.0)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--opacity", type=float, default=0.05)
    a = p.parse_args()

    import mujoco
    import torch
    from gsplat import rasterization
    from plyfile import PlyData

    dev = "cuda"
    H = a.height

    # ---- the scene side: mesh depth and the shared-surface mask -------------
    model = mujoco.MjModel.from_xml_path(str(a.scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    ci = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cap")
    Ccap = data.cam_xpos[ci].copy()
    Rocv = data.cam_xmat[ci].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    fovy = float(model.cam_fovy[ci])
    W = int(round(H * 1080 / 1920))

    shared_bodies = ("obj0_round_wooden_table", "obj1_red_apple",
                     "obj2_woven_brown_basket")
    gids = []
    for nm in shared_bodies:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        gids += [g for g in range(model.ngeom)
                 if int(model.geom_bodyid[g]) == b and int(model.geom_group[g]) != 3]

    r = mujoco.Renderer(model, H, W)
    vopt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(vopt)
    vopt.sitegroup[:] = 0
    r.enable_segmentation_rendering()
    r.update_scene(data, camera="cap", scene_option=vopt)
    seg = r.render()[:, :, 0].astype(np.int32)
    r.disable_segmentation_rendering()
    r.enable_depth_rendering()
    r.update_scene(data, camera="cap", scene_option=vopt)
    mesh_depth = r.render().astype(np.float64)
    r.disable_depth_rendering()
    r.close()
    mask = np.isin(seg, gids)
    print(f"[fit] shared-surface mask: {mask.sum()} px ({100*mask.mean():.1f}% of "
          f"{W}x{H}) over table + apple + basket", flush=True)
    print(f"[fit] mesh depth there: {mesh_depth[mask].min():.3f}..."
          f"{mesh_depth[mask].max():.3f} m", flush=True)

    # ---- the splat, on the GPU once ----------------------------------------
    v = PlyData.read(str(a.ply))["vertex"]
    keep = (1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32)))) > a.opacity
    means = np.stack([v["x"], v["y"], v["z"]], 1)[keep].astype(np.float32)
    T = [torch.from_numpy(x).to(dev) for x in (
        means,
        np.stack([v[f"rot_{i}"] for i in range(4)], 1)[keep].astype(np.float32),
        np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1)[keep]).astype(np.float32),
        (1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32))))[keep].astype(np.float32),
        np.clip(0.5 + 0.2820948 * np.stack([v[f"f_dc_{i}"] for i in range(3)], 1)[keep],
                0, 1).astype(np.float32))]
    print(f"[fit] {len(means):,} gaussians", flush=True)

    f = (H / 2) / np.tan(np.deg2rad(fovy) / 2)
    K = torch.tensor([[[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]]],
                     dtype=torch.float32, device=dev)

    poses = json.load(open(a.poses))
    C = np.array([poses[k]["C"] for k in poses])
    up = np.linalg.svd(C - C.mean(0), full_matrices=False)[2][2]
    if (means[::97] @ up < np.median(C @ up)).mean() < 0.5:
        up = -up
    R0 = basis_from_up(up)
    b1 = np.cross(up, [0.0, 0.0, 1.0]); b1 /= np.linalg.norm(b1)
    b2 = np.cross(up, b1)
    anchor = a.table_xy[0] * b1 + a.table_xy[1] * b2 + a.table_h * up
    wt = np.asarray(a.world_table, float)

    def splat_depth(scale, yaw):
        R = yaw_mat(yaw) @ R0
        t = wt - scale * (R @ anchor)
        Cs = (R.T / scale) @ (Ccap - t)
        Rcs = R.T @ Rocv
        w2c = np.eye(4)
        w2c[:3, :3] = Rcs.T
        w2c[:3, 3] = -Rcs.T @ Cs
        vm = torch.from_numpy(w2c.astype(np.float32))[None].to(dev)
        with torch.no_grad():
            out, alpha, _ = rasterization(*T, vm, K, W, H, packed=False,
                                          render_mode="RGB+ED")
        d = out[0, :, :, 3].cpu().numpy().astype(np.float64)
        al = alpha[0, :, :, 0].cpu().numpy()
        # depth comes back in splat units; world depth is scale * that, which is
        # exactly why scale is observable here and not in the image
        return scale * d, al

    def score(scale, yaw):
        d, al = splat_depth(scale, yaw)
        m = mask & (al > 0.5) & (d > 1e-3)
        if m.sum() < 0.2 * mask.sum():
            return None
        diff = d[m] - mesh_depth[m]
        return dict(n=int(m.sum()), med=float(np.median(diff)),
                    mad=float(np.median(np.abs(diff - np.median(diff)))),
                    p3=float((np.abs(diff) < 0.03).mean()),
                    p10=float((np.abs(diff) < 0.10).mean()))

    los, his, st = a.scale_range
    print(f"\n[fit] coarse grid: scale {los}..{his} step {st}, yaw step {a.yaw_step}",
          flush=True)
    best = None
    for sc in np.arange(los, his + 1e-9, st):
        row = []
        for yaw in np.arange(0.0, 360.0, a.yaw_step):
            s_ = score(float(sc), float(yaw))
            if s_ is None:
                continue
            row.append((s_["p3"], float(yaw), s_))
            if best is None or s_["p3"] > best[0]:
                best = (s_["p3"], float(sc), float(yaw), s_)
        if row:
            row.sort(reverse=True)
            print(f"  scale {sc:.3f}: best yaw {row[0][1]:6.1f}  "
                  f"within3cm {100*row[0][0]:5.1f}%  within10cm "
                  f"{100*row[0][2]['p10']:5.1f}%  median {row[0][2]['med']:+.3f} m",
                  flush=True)
    if best is None:
        raise SystemExit("no candidate produced enough coverage on the shared mask")
    print(f"\n[fit] coarse best: scale {best[1]:.3f} yaw {best[2]:.1f} -> "
          f"within3cm {100*best[0]:.1f}%", flush=True)

    for ds, dy in ((0.01, 3.0), (0.0025, 0.75)):
        sc0, yw0 = best[1], best[2]
        for sc in np.arange(sc0 - 2 * ds, sc0 + 2 * ds + 1e-9, ds):
            for yaw in np.arange(yw0 - 2 * dy, yw0 + 2 * dy + 1e-9, dy):
                s_ = score(float(sc), float(yaw))
                if s_ and s_["p3"] > best[0]:
                    best = (s_["p3"], float(sc), float(yaw), s_)
        print(f"  refined (+-{ds}, +-{dy} deg) -> scale {best[1]:.4f} "
              f"yaw {best[2]:.2f}  within3cm {100*best[0]:.1f}%", flush=True)

    st_ = best[3]
    print(f"\n[fit] BEST scale {best[1]:.4f}  yaw {best[2]:.2f} deg")
    print(f"  within 3 cm {100*st_['p3']:.1f}%   within 10 cm {100*st_['p10']:.1f}%")
    print(f"  median depth error {st_['med']:+.4f} m   MAD {st_['mad']:.4f} m")
    print(f"  (bg_lm.ply, for comparison: 19.6% within 3 cm, 39.1% within 10 cm)")
    a.out.write_text(json.dumps(dict(scale=best[1], yaw_deg=best[2],
                                     stats=st_), indent=1))
    print(f"[fit] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
