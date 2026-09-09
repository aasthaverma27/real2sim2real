"""Preview an unpacked 3DGS room/scene PLY from inside.

Camera is placed at the scene centroid, looking outward along the
four cardinal directions (+x, -x, +y, -y) and slightly down. Splats
use a vectorised additive blend — fast enough for ~500K gaussians.

Sanity-check only — full quality needs gsplat on GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData

SH_C0 = 0.28209479177387814


def load(p: Path) -> dict[str, np.ndarray]:
    v = PlyData.read(p)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    sh_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)
    color = np.clip(0.5 + SH_C0 * sh_dc, 0, 1)
    opacity = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32)))
    return {"xyz": xyz, "color": color, "opacity": opacity}


def look_at(eye, target, up=np.array([0, 0, 1.0], dtype=np.float32)):
    f = target - eye
    f /= np.linalg.norm(f) + 1e-12
    s = np.cross(f, up)
    s /= np.linalg.norm(s) + 1e-12
    u = np.cross(s, f)
    return np.stack([s, u, -f], 0)


def render(g, eye, target, W=480, H=320, fov_deg=70, splat_r=1):
    R = look_at(eye, target)
    pts = (g["xyz"] - eye) @ R.T
    z = -pts[:, 2]
    valid = z > 0.05
    pts, z = pts[valid], z[valid]
    color = g["color"][valid]
    op = g["opacity"][valid]
    fy = np.deg2rad(fov_deg)
    f = (H / 2.0) / np.tan(fy / 2.0)
    u = pts[:, 0] * f / z + W / 2.0
    v = -pts[:, 1] * f / z + H / 2.0

    order = np.argsort(-z)
    u, v, color, op = u[order].astype(np.int32), v[order].astype(np.int32), color[order], op[order]

    in_screen = (u >= splat_r) & (u < W - splat_r) & (v >= splat_r) & (v < H - splat_r)
    u, v, color, op = u[in_screen], v[in_screen], color[in_screen], op[in_screen]

    accum = np.zeros((H, W, 3), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)
    for dy in range(-splat_r, splat_r + 1):
        for dx in range(-splat_r, splat_r + 1):
            falloff = float(np.exp(-(dx * dx + dy * dy) / max(1.0, splat_r)))
            w = op * falloff
            np.add.at(accum, (v + dy, u + dx), color * w[:, None])
            np.add.at(wsum, (v + dy, u + dx), w)
    mask = wsum > 1e-6
    img = np.full((H, W, 3), 0.05, dtype=np.float32)
    img[mask] = accum[mask] / wsum[mask, None]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def label(np_img, txt):
    im = Image.fromarray(np_img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 18], fill=(0, 0, 0))
    d.text((4, 2), txt, fill=(255, 255, 255))
    return np.array(im)


def make_panel(name: str, ply: Path, out: Path):
    g = load(ply)
    centroid = g["xyz"].mean(0)
    # bounding-box-aware "near floor" eye: 30% of height above floor
    zmin = g["xyz"][:, 2].min()
    zmax = g["xyz"][:, 2].max()
    eye_z = zmin + 0.4 * (zmax - zmin)
    eye = np.array([centroid[0], centroid[1], eye_z], dtype=np.float32)
    ext = g["xyz"].max(0) - g["xyz"].min(0)

    cols = []
    for tag, dx, dy in [("+x", 1, 0), ("-x", -1, 0), ("+y", 0, 1), ("-y", 0, -1)]:
        target = eye + np.array([dx * ext[0], dy * ext[1], -0.05 * ext[2]], dtype=np.float32)
        img = render(g, eye, target)
        cols.append(label(img, f"{name} · look {tag} · n={g['xyz'].shape[0]}"))
    row = np.concatenate(cols, axis=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(row).save(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plys", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    rows = []
    for name, ply in zip(args.names, args.plys):
        tmp = args.out.with_suffix(f".{name}.png")
        make_panel(name, Path(ply), tmp)
        rows.append(np.array(Image.open(tmp)))
        tmp.unlink()
    grid = np.concatenate(rows, axis=0)
    Image.fromarray(grid).save(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
