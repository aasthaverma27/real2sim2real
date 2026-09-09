"""Decompress SuperSplat / PlayCanvas packed 3DGS PLY → official 3DGS PLY.

DISCOVERSE assets are stored in the PlayCanvas compressed format
(`element chunk` + `element vertex` with uint32 packed_* fields).
This script converts a single file or directory of such PLYs into
the standard 3DGS layout (`x/y/z`, `f_dc_*`, `scale_*`, `rot_*`,
`opacity`) that `mugs.sensors.GaussianSensor._load_official_ply`
already understands.

Bit layout (per playcanvas engine `gsplat-compressed-data.js`):
  packed_position : uint32   11/10/11 bits  → lerp(chunk min,max)
  packed_scale    : uint32   11/10/11 bits  → lerp(chunk min,max)  (log-space)
  packed_rotation : uint32   2/10/10/10     → smallest-three quat
  packed_color    : uint32   8/8/8/8        → R,G,B,A in [0,1]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


SH_C0 = 0.28209479177387814


def _unpack_unorm(values: np.ndarray, shift: int, bits: int) -> np.ndarray:
    mask = (1 << bits) - 1
    return ((values >> shift) & mask).astype(np.float64) / float(mask)


def _smallest_three_quat(packed: np.ndarray) -> np.ndarray:
    sel = (packed >> 30) & 0x3
    a = _unpack_unorm(packed, 20, 10) - 0.5
    b = _unpack_unorm(packed, 10, 10) - 0.5
    c = _unpack_unorm(packed, 0, 10) - 0.5
    s = math.sqrt(2.0)
    a *= s
    b *= s
    c *= s
    d2 = 1.0 - (a * a + b * b + c * c)
    d = np.sqrt(np.clip(d2, 0.0, None))

    quats = np.zeros((packed.shape[0], 4), dtype=np.float64)
    # Convention used by playcanvas engine: ordering [x, y, z, w], the
    # selector indexes the *largest* magnitude component. Official 3DGS
    # PLYs store rot_0..rot_3 as [w, x, y, z], so map accordingly.
    # playcanvas index 0->x, 1->y, 2->z, 3->w  -->  official wxyz idx
    pc_to_wxyz = np.array([1, 2, 3, 0])
    for pc_i in range(4):
        mask = sel == pc_i
        if not mask.any():
            continue
        out_idx = pc_to_wxyz[pc_i]
        quats[mask, out_idx] = d[mask]
        remaining_pc = [j for j in range(4) if j != pc_i]
        remaining_out = pc_to_wxyz[remaining_pc]
        quats[mask, remaining_out[0]] = a[mask]
        quats[mask, remaining_out[1]] = b[mask]
        quats[mask, remaining_out[2]] = c[mask]

    quats /= np.linalg.norm(quats, axis=1, keepdims=True) + 1e-12
    return quats


def decompress_supersplat(plydata: PlyData) -> dict[str, np.ndarray]:
    if "chunk" not in {e.name for e in plydata.elements}:
        raise ValueError("not a compressed SuperSplat PLY (missing 'chunk' element)")

    chunks = plydata["chunk"]
    verts = plydata["vertex"]
    n = len(verts)
    chunk_size = 256
    chunk_idx = np.minimum(np.arange(n) // chunk_size, len(chunks) - 1)

    def per_vertex(field: str) -> np.ndarray:
        return chunks[field][chunk_idx]

    pmin = np.stack([per_vertex("min_x"), per_vertex("min_y"), per_vertex("min_z")], 1)
    pmax = np.stack([per_vertex("max_x"), per_vertex("max_y"), per_vertex("max_z")], 1)
    smin = np.stack(
        [per_vertex("min_scale_x"), per_vertex("min_scale_y"), per_vertex("min_scale_z")],
        1,
    )
    smax = np.stack(
        [per_vertex("max_scale_x"), per_vertex("max_scale_y"), per_vertex("max_scale_z")],
        1,
    )

    pp = verts["packed_position"].astype(np.uint32)
    pn = np.stack(
        [_unpack_unorm(pp, 21, 11), _unpack_unorm(pp, 11, 10), _unpack_unorm(pp, 0, 11)],
        1,
    )
    positions = pmin + pn * (pmax - pmin)

    ps = verts["packed_scale"].astype(np.uint32)
    sn = np.stack(
        [_unpack_unorm(ps, 21, 11), _unpack_unorm(ps, 11, 10), _unpack_unorm(ps, 0, 11)],
        1,
    )
    log_scales = smin + sn * (smax - smin)

    quats_wxyz = _smallest_three_quat(verts["packed_rotation"].astype(np.uint32))

    pc = verts["packed_color"].astype(np.uint32)
    r = _unpack_unorm(pc, 24, 8)
    g = _unpack_unorm(pc, 16, 8)
    b = _unpack_unorm(pc, 8, 8)
    alpha = _unpack_unorm(pc, 0, 8)
    color_bound_fields = chunks.data.dtype.names
    if "min_r" in color_bound_fields:
        rmin = per_vertex("min_r")
        rmax = per_vertex("max_r")
        gmin = per_vertex("min_g")
        gmax = per_vertex("max_g")
        bmin = per_vertex("min_b")
        bmax = per_vertex("max_b")
        r = rmin + r * (rmax - rmin)
        g = gmin + g * (gmax - gmin)
        b = bmin + b * (bmax - bmin)
    colors_rgb = np.stack([r, g, b], 1)
    colors_rgb = np.clip(colors_rgb, 1e-6, 1.0 - 1e-6)
    alpha = np.clip(alpha, 1e-6, 1.0 - 1e-6)

    # Convert color back to SH DC (the official format stores SH DC,
    # then renderer does 0.5 + SH_C0 * dc). Inverse:
    f_dc = (colors_rgb - 0.5) / SH_C0
    # Opacity is stored pre-sigmoid in official format; renderer does sigmoid.
    opacity_pre = np.log(alpha / (1.0 - alpha))

    return {
        "xyz": positions.astype(np.float32),
        "f_dc": f_dc.astype(np.float32),
        "scales_log": log_scales.astype(np.float32),
        "rot_wxyz": quats_wxyz.astype(np.float32),
        "opacity_pre": opacity_pre.astype(np.float32),
    }


def write_official_ply(out_path: Path, gaussians: dict[str, np.ndarray]) -> None:
    n = gaussians["xyz"].shape[0]
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    arr = np.zeros(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = gaussians["xyz"].T
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = gaussians["f_dc"].T
    arr["opacity"] = gaussians["opacity_pre"]
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = gaussians["scales_log"].T
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = gaussians["rot_wxyz"].T
    el = PlyElement.describe(arr, "vertex")
    PlyData([el], byte_order="<").write(str(out_path))


def convert_file(src: Path, dst: Path) -> dict:
    src = src.resolve()
    plydata = PlyData.read(src)
    g = decompress_supersplat(plydata)
    dst.parent.mkdir(parents=True, exist_ok=True)
    write_official_ply(dst, g)
    bbox = g["xyz"].max(0) - g["xyz"].min(0)
    return {
        "src": str(src),
        "dst": str(dst),
        "n": g["xyz"].shape[0],
        "bbox_m": tuple(round(float(x), 3) for x in bbox),
        "center_m": tuple(round(float(x), 3) for x in g["xyz"].mean(0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="Compressed ply file or directory")
    ap.add_argument("dst", type=Path, help="Output ply file or directory")
    args = ap.parse_args()

    if args.src.is_dir():
        for ply in sorted(args.src.glob("*.ply")):
            info = convert_file(ply, args.dst / ply.name)
            print(info)
    else:
        info = convert_file(args.src, args.dst)
        print(info)


if __name__ == "__main__":
    main()
