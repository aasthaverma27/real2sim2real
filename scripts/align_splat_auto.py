#!/usr/bin/env python3
"""Place a COLMAP splat in the scene's metric frame, without anyone's eyes.

The interactive tuner exists because five automated objectives that fitted the
splat's *geometry* against the mesh gave five different yaws
(`docs/real2sim_experiments.md` F14). This does not fit geometry at all. It uses
the one thing the two reconstructions already share:

    the object still IS frame 0 of the room video

so the very same physical camera appears in both, and a camera pose is six
degrees of freedom pinned down exactly. Nothing is being optimised, so there is
nothing for competing objectives to disagree about.

What comes from where:

    COLMAP        image 00001's pose        -> that camera in the splat's frame
    gravity_alignment.json                  -> the same camera in the object frame
    rescale_summary.json                    -> object frame -> metres

Rotation and camera position follow from those directly, and exactly. Scale does
not - a single pose says nothing about how large COLMAP's units are.

Scale comes from the splat itself, against the one length that was actually
measured with a tape. Once the rotation above has made the splat level, rays are
cast from the frame-0 camera through floor pixels and through the table's own
mask, which gives the height of the floor and of the table top *in the splat*.
Their difference is the table, in COLMAP units; the measured table height in
metres divides out, and that is the scale.

Deriving it instead from VGGT's depth map - the obvious thing, and what this
script did first - is wrong by a factor of 2.6 here. VGGT stretches along the
viewing ray (see rescale_scene.py), so the depths it reports for frame 0 are not
in the same proportion as COLMAP's, and a median over thousands of pixels is a
confident average of an inconsistent quantity. Two horizontal planes and a tape
measure have no such problem.

    python scripts/align_splat_auto.py --scene-dir outputs/myroom

Writes `<scene>/splat_transform.json`, which `apply_splat_transform.py
--transform` consumes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None,
                   help="default <scene-dir>/splat_transform.json")
    p.add_argument("--table-height", type=float, required=True,
                   help="The measured table height, in metres - the same number "
                        "the rescale stage was anchored on. This is what makes "
                        "the splat metric.")
    p.add_argument("--table-index", type=int, default=0,
                   help="Which sam3d/N.png is the table.")
    p.add_argument("--ply", type=Path, default=None,
                   help="default <scene-dir>/gs_wide.ply - the UNtransformed splat")
    p.add_argument("--min-opacity", type=float, default=0.2)
    p.add_argument("--image-width", type=int, default=1080)
    p.add_argument("--image-height", type=int, default=1920,
                   help="Resolution the sam3d masks are in, i.e. the object "
                        "image's - the colmap intrinsics are scaled to it.")
    p.add_argument("--max-spread", type=float, default=0.02,
                   help="Refuse if the scale is pinned to worse than this "
                        "fraction. This is the uncertainty of the median, not "
                        "the scatter of individual points - a wrong scale is "
                        "not visible in the numbers, only in the render.")
    a = p.parse_args()

    import pycolmap
    from plyfile import PlyData

    scene = a.scene_dir
    out = a.out or (scene / "splat_transform.json")

    # ---- the same camera, in both reconstructions ------------------------
    rec = pycolmap.Reconstruction(str(scene / "roommesh" / "colmap" / "sparse" / "0"))
    first = min(rec.images.values(), key=lambda i: i.name)
    print(f"[auto-align] frame 0 is {first.name} in COLMAP, "
          f"{len(rec.images)} images total")
    w2c = first.cam_from_world()
    R_c = w2c.rotation.matrix()                 # world(colmap) -> camera
    C_col = -R_c.T @ w2c.translation            # camera centre, colmap units

    cam = rec.cameras[first.camera_id]

    ga = json.loads((scene / "gravity_alignment.json").read_text())
    T_gc = np.asarray(ga["transform_gravity_from_camera"], float)
    if abs(np.linalg.det(T_gc[:3, :3]) - 1.0) > 1e-3:
        raise SystemExit("transform_gravity_from_camera is not a rigid motion; "
                         "this derivation assumes it carries no scale")
    R_gc, c_g = T_gc[:3, :3], T_gc[:3, 3]

    rs = json.loads((scene / "mujoco" / "rescale_summary.json").read_text())
    factor, floor_z = float(rs["factor"]), float(rs["floor_z"])

    # colmap world -> scene, rotation only. The yaw in this is right - it comes
    # from the camera the two reconstructions share - but the LEVEL in it does
    # not: it traces back to a plane fitted to one VGGT depth map, and in this
    # scene that plane is 6.4 degrees off. The splat's own floor, with tens of
    # thousands of points on it, settles that far better, so the tilt gets
    # corrected below and only the yaw is kept.
    R = R_gc @ R_c

    # ---- level it on its own floor --------------------------------------
    from PIL import Image

    ply = PlyData.read(str(a.ply or (scene / "gs_wide.ply")))["vertex"]
    Pc = np.stack([ply["x"], ply["y"], ply["z"]], 1).astype(np.float64)
    opa = 1.0 / (1.0 + np.exp(-ply["opacity"].astype(np.float64)))
    Pc = Pc[opa > a.min_opacity]
    print(f"[auto-align] {len(Pc):,} solid gaussians")

    reach = float(np.median(np.linalg.norm(Pc - C_col, axis=1)))

    def floor_plane(Rot):
        """RANSAC the dominant near-level plane of the splat, seen through Rot."""
        Q = (Pc - C_col) @ Rot.T
        Q = Q[(Q[:, 2] > -1.6 * reach) & (Q[:, 2] < 0.25 * reach)]
        Q = Q[np.linalg.norm(Q[:, :2], axis=1) < 1.5 * reach]
        if len(Q) < 2000:
            return None
        rng = np.random.default_rng(0)
        best = None
        tol = 0.012 * reach
        for _ in range(3000):
            t3 = Q[rng.integers(0, len(Q), 3)]
            n = np.cross(t3[1] - t3[0], t3[2] - t3[0])
            L = np.linalg.norm(n)
            if L < 1e-12:
                continue
            n = n / L
            if abs(n[2]) < 0.85:          # the floor, not a wall
                continue
            cnt = int((np.abs((Q - t3[0]) @ n) < tol).sum())
            if best is None or cnt > best[0]:
                best = (cnt, n if n[2] > 0 else -n, t3[0])
        if best is None:
            return None
        cnt, n, p0 = best
        inl = Q[np.abs((Q - p0) @ n) < tol]
        return cnt, n, float(np.median(inl @ n))

    fp = floor_plane(R)
    if fp is None:
        raise SystemExit("no floor plane in the splat; fall back to the tuner")
    cnt, n, _ = fp
    tilt = float(np.degrees(np.arccos(np.clip(n[2], -1, 1))))
    if tilt > 0.05:
        ax = np.cross(n, [0.0, 0.0, 1.0])
        L = np.linalg.norm(ax)
        if L > 1e-9:
            ax = ax / L
            th = np.arccos(np.clip(n[2], -1, 1))
            Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            R = (np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)) @ R
    print(f"[auto-align] floor plane: {cnt:,} inliers, was {tilt:.2f} deg off "
          f"level; corrected (yaw kept)")
    fp = floor_plane(R)
    z_floor = fp[2]
    print(f"[auto-align] residual tilt {np.degrees(np.arccos(np.clip(fp[1][2],-1,1))):.2f} deg")

    # ---- scale: the splat's own table, against the measured one -----------
    # Intrinsics from COLMAP, not from VGGT: these were fitted by bundle
    # adjustment over 450 views that all had to agree, and they differ from
    # VGGT's guess for the same camera by 11% - which is 11% of every angle a
    # ray below is cast at.
    sx, sy = a.image_width / cam.width, a.image_height / cam.height
    fx, fy = cam.params[0] * sx, cam.params[1] * sy
    ux, uy = cam.params[2] * sx, cam.params[3] * sy
    print(f"[auto-align] colmap intrinsics at {a.image_width}x{a.image_height}: "
          f"fx={fx:.1f} fy={fy:.1f}")

    Rc2w = R_c.T

    def cast(u, v, frac=0.012):
        d = Rc2w @ np.array([(u - ux) / fx, (v - uy) / fy, 1.0])
        d = d / np.linalg.norm(d)
        w = Pc - C_col
        t = w @ d
        perp = np.linalg.norm(w - np.outer(t, d), axis=1)
        for f in (frac, 2 * frac, 4 * frac):
            m = (perp < f * reach) & (t > 0.02 * reach)
            if m.sum() >= 40:
                hit = C_col + float(np.percentile(t[m], 5)) * d
                return float(((hit - C_col) @ R.T)[2])
        return None

    masks = sorted((scene / "sam3d").glob("*.png"), key=lambda q: int(q.stem))
    alpha = np.array(Image.open(masks[a.table_index]).convert("RGBA"))[..., 3] > 128
    ys, xs = np.nonzero(alpha)
    cut = ys.min() + 0.35 * (ys.max() - ys.min())
    sel = ys < cut
    z_top = cast(xs[sel].mean(), ys[sel].mean())
    if z_top is None:
        raise SystemExit("the table's own mask hits nothing in the splat")
    h_col = z_top - z_floor
    if h_col <= 1e-9:
        raise SystemExit(f"table came out below the floor ({h_col:.4f})")
    s = a.table_height / h_col
    print(f"[auto-align] splat table is {h_col:.4f} colmap units above its floor, "
          f"measured {a.table_height:.3f} m -> scale {s:.5f}")

    # ---- compose ---------------------------------------------------------
    # Rotation and the camera's position come from the object frame; scale comes
    # from the splat. Anchor the translation on the camera, because that is the
    # quantity the two reconstructions actually share, then drop the whole thing
    # so the splat's floor - not the object frame's idea of the floor - is z=0.
    cam_scene = factor * (c_g - np.array([0.0, 0.0, floor_z]))
    t = cam_scene - s * (R @ C_col)
    floor_now = float(s * z_floor + (cam_scene - s * (R @ C_col))[2])
    t = t - np.array([0.0, 0.0, floor_now])
    print(f"[auto-align] splat floor was at z={floor_now:+.3f} m; dropped to 0")

    cam_scene = factor * (c_g - np.array([0.0, 0.0, floor_z]))
    yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    print(f"[auto-align] scale {s:.5f}   yaw {yaw:+.2f} deg")
    print(f"[auto-align] frame 0 camera stands at "
          f"({cam_scene[0]:+.3f}, {cam_scene[1]:+.3f}, {cam_scene[2]:+.3f}) m")

    # Independent check, and the only one worth making: push EVERY colmap camera
    # through the transform. It was a person walking around a room holding a
    # phone, so the answer has to be a roughly level path at roughly chest
    # height. If gravity came out wrong the path tilts, and that shows up here
    # as a height spread of a metre rather than a few centimetres. (Comparing
    # the two frames' z axes instead would prove nothing - colmap's z is
    # whatever the first camera happened to be looking along.)
    C_all = np.stack([-i.cam_from_world().rotation.matrix().T
                      @ i.cam_from_world().translation
                      for i in rec.images.values()])
    P = (s * (C_all @ R.T) + t)
    h_mid, h_sd = float(np.median(P[:, 2])), float(np.std(P[:, 2]))
    span = P[:, :2].max(0) - P[:, :2].min(0)
    print(f"[auto-align] camera path: height {h_mid:.2f} m +/-{h_sd:.2f}, "
          f"covering {span[0]:.1f} x {span[1]:.1f} m of floor")
    level = h_sd < 0.25 and 0.8 < h_mid < 2.0
    if not level:
        print("[auto-align] WARNING: that is not a person walking around a room. "
              "A height spread over ~0.25 m means the splat is not level, and a "
              "median outside 0.8-2.0 m means the scale is wrong. Check the "
              "render before trusting this, and fall back to the interactive "
              "tuner if it looks wrong.")

    out.write_text(json.dumps({
        "R": R.tolist(), "s": s, "t": t.tolist(),
        "source": "align_splat_auto.py: frame-0 camera shared by COLMAP and VGGT",
        "colmap_image": first.name,
        "table_height_m": a.table_height,
        "splat_table_height_colmap_units": h_col,
        "floor_tilt_corrected_deg": tilt,
        "camera_rotation_in_scene": (R @ R_c.T).tolist(),
        "intrinsics_at_mask_res": [fx, fy, ux, uy],
        "camera_in_scene": cam_scene.tolist(),
        "yaw_deg": yaw,
        "camera_path_height_m": h_mid,
        "camera_path_height_sd_m": h_sd,
        "camera_path_span_m": span.tolist(),
    }, indent=2))
    print(f"[auto-align] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
