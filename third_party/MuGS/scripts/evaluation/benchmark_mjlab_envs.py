#!/usr/bin/env python3
"""End-to-end mjlab + MuGS GaussianSensor multi-env benchmark.

Thin CLI shim around `mugs_mjlab.tasks` so anyone can reproduce the
README throughput tables (or sub in their own PLY) with a single command.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import torch

from mugs_mjlab.tasks import (
    benchmark_env_cfg,
    kitchen_ply_path,
    make_yam_lift_cube_gs_env_cfg,
    print_benchmark_table,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ply",
        type=Path,
        default=None,
        help="3DGS PLY background path (defaults to bundled kitchen scene).",
    )
    p.add_argument("--width", type=int, default=160)
    p.add_argument("--height", type=int, default=120)
    p.add_argument(
        "--render-mode",
        choices=("hybrid", "3dgs_only", "mujoco_only"),
        default="hybrid",
    )
    p.add_argument(
        "--env-counts",
        type=int,
        nargs="+",
        default=[1, 4, 16, 64, 256, 1024, 4096],
        help="Batch sizes to sweep.",
    )
    p.add_argument("--n-steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ply = args.ply or kitchen_ply_path()
    assert ply.exists(), f"ply missing: {ply}"

    print(f"Device: {torch.cuda.get_device_name(0)}")
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"Free/Total VRAM: {free_b / 1e9:.1f}/{total_b / 1e9:.1f} GB\n")

    def factory(num_envs: int):
        return make_yam_lift_cube_gs_env_cfg(
            ply_path=ply,
            num_envs=num_envs,
            width=args.width,
            height=args.height,
            render_mode=args.render_mode,
        )

    title = (
        f"mjlab YAM-lift + GaussianSensorMjlab @ "
        f"{args.width}×{args.height}, render_mode={args.render_mode}"
    )
    results = benchmark_env_cfg(
        factory,
        env_counts=tuple(args.env_counts),
        n_steps=args.n_steps,
        warmup=args.warmup,
    )
    print_benchmark_table(results, title=title)


if __name__ == "__main__":
    main()
