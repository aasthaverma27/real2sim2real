"""Render a quick CPU preview of an unpacked 3DGS PLY to verify decompression.

This is a sanity-check renderer (depth-sorted point splat with the
official 3DGS attribute decoding: f_dc → SH DC → RGB, opacity → sigmoid,
scale → exp). It does NOT do the full anisotropic 2D Gaussian projection
— but for objects of ~10cm rendered from ~40cm it produces a clearly
recognisable silhouette good enough to confirm the file is real.

Output is a single PNG showing front / side / top orbit views.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData

SH_C0 = 0.28209479177387814


def load_official_ply(p: Path) -> dict[str, np.ndarray]:
    v = PlyData.read(p)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    sh_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)
    colors = np.clip(0.5 + SH_C0 * sh_dc, 0, 1)
    opacity = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32)))
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float32))
    return {"xyz": xyz, "colors": colors, "opacity": opacity, "scales": scales}


def look_at(eye, target, up):
    f = target - eye
    f /= np.linalg.norm(f) + 1e-12
    s = np.cross(f, up)
    s /= np.linalg.norm(s) + 1e-12
    u = np.cross(s, f)
    R = np.stack([s, u, -f], 0)  # world → cam, OpenGL convention (cam looks -z)
    return R


def project(xyz, eye, R, fy, W, H):
    pts = (xyz - eye) @ R.T
    z = -pts[:, 2]
    valid = z > 0.01
    pts = pts[valid]
    z = z[valid]
    f = (H / 2.0) / np.tan(fy / 2.0)
    u = pts[:, 0] * f / z + W / 2.0
    v = -pts[:, 1] * f / z + H / 2.0
    return u, v, z, valid, f


def splat(image, depth_buf, u, v, z, color, opacity, radius_px):
    H, W = image.shape[:2]
    order = np.argsort(-z)  # back-to-front for alpha-over
    u = u[order].astype(np.int32)
    v = v[order].astype(np.int32)
    color = color[order]
    opacity = opacity[order]
    in_screen = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, color, opacity = u[in_screen], v[in_screen], color[in_screen], opacity[in_screen]
    r = radius_px
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d2 = dx * dx + dy * dy
            if d2 > r * r:
                continue
            w = np.exp(-d2 / max(1.0, r * 0.5)) * opacity
            yy = np.clip(v + dy, 0, H - 1)
            xx = np.clip(u + dx, 0, W - 1)
            # vectorized alpha-over per pixel via np.add.at would overdraw —
            # accept slight artefacts from duplicate writes; loop dy/dx is plenty for a preview
            blend = w[:, None]
            image[yy, xx] = image[yy, xx] * (1 - blend) + color * blend


def render_view(g, eye, target, W=320, H=320, fov_deg=35):
    R = look_at(eye, target, np.array([0, 0, 1.0], dtype=np.float32))
    fy = np.deg2rad(fov_deg)
    u, v, z, valid, _ = project(g["xyz"], eye, R, fy, W, H)
    img = np.ones((H, W, 3), dtype=np.float32) * 0.97
    depth = np.full((H, W), np.inf, dtype=np.float32)
    splat(img, depth, u, v, z, g["colors"][valid], g["opacity"][valid], radius_px=2)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def labeled(img_np, label):
    im = Image.fromarray(img_np)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 18], fill=(0, 0, 0))
    d.text((4, 2), label, fill=(255, 255, 255))
    return np.array(im)


def make_grid(plys: list[Path], out: Path):
    rows = []
    for p in plys:
        g = load_official_ply(p)
        center = g["xyz"].mean(0)
        radius = float(np.linalg.norm(g["xyz"] - center, axis=1).max()) * 4.5
        radius = max(radius, 0.25)
        views = []
        for label, eye in [
            ("front", center + np.array([0, -radius, 0])),
            ("side", center + np.array([radius, 0, 0])),
            ("top", center + np.array([0, 0, radius])),
        ]:
            img = render_view(g, eye, center)
            views.append(labeled(img, f"{p.stem} · {label} · n={g['xyz'].shape[0]}"))
        rows.append(np.concatenate(views, axis=1))
    grid = np.concatenate(rows, axis=0)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plys", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = make_grid(args.plys, args.out)
    print("wrote", out)


if __name__ == "__main__":
    main()
