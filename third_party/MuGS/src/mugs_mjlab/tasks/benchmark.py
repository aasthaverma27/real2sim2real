"""Reusable env-FPS benchmark runner for mjlab + MuGS sensors.

Generic over the env config — pass any factory `int -> ManagerBasedRlEnvCfg`
and `benchmark_env_cfg` will sweep batch sizes, time `env.step()`, and report
peak VRAM. Used to produce the throughput tables in the README.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnvCfg

EnvCfgFactory = Callable[[int], "ManagerBasedRlEnvCfg"]


@dataclass
class BenchmarkResult:
    """One row of a batch-size sweep."""

    num_envs: int
    step_latency_s: float | None
    """Mean wall-clock seconds per `env.step()`. None if the run errored."""

    env_fps: float | None
    """num_envs / step_latency. None if the run errored."""

    peak_vram_gb: float | None
    """Peak CUDA allocator usage in GB during this run. None on error."""

    error: str | None = None
    """Short error message (OOM / runtime exception). None on success."""


def _build_env(env_cfg: "ManagerBasedRlEnvCfg", device: str):
    # Local import keeps the module importable even when mjlab is not
    # installed (e.g. for type-checking on a CPU-only machine).
    from mjlab.envs import ManagerBasedRlEnv

    return ManagerBasedRlEnv(cfg=env_cfg, device=device)


def benchmark_env_cfg(
    env_cfg_factory: EnvCfgFactory,
    *,
    env_counts: Sequence[int] = (1, 4, 16, 64, 256, 1024, 4096),
    n_steps: int = 30,
    warmup: int = 5,
    device: str = "cuda",
) -> list[BenchmarkResult]:
    """Sweep batch sizes and measure mean per-step latency + peak VRAM.

    Drives each env with zero-actions so the result reflects raw pipeline
    cost (physics + sensors + obs/reward managers), not policy compute.

    Args:
        env_cfg_factory: `num_envs -> ManagerBasedRlEnvCfg`. Called once per
            batch size; should return a freshly constructed config.
        env_counts: Batch sizes to sweep, smallest first.
        n_steps: Number of timed steps per batch size (after warmup).
        warmup: Number of un-timed warmup steps (lets CUDA graphs build).
        device: Torch device string passed to `ManagerBasedRlEnv`.

    Returns:
        One `BenchmarkResult` per entry in `env_counts`, in order.
    """
    results: list[BenchmarkResult] = []
    for num_envs in env_counts:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        env = None
        try:
            env = _build_env(env_cfg_factory(num_envs), device=device)
            env.reset()
            action = torch.zeros(env.action_space.shape, device=device)
            for _ in range(warmup):
                env.step(action)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n_steps):
                env.step(action)
            torch.cuda.synchronize()
            dt = (time.time() - t0) / n_steps
            peak = torch.cuda.max_memory_allocated() / 1e9
            results.append(
                BenchmarkResult(
                    num_envs=num_envs,
                    step_latency_s=dt,
                    env_fps=num_envs / dt,
                    peak_vram_gb=peak,
                )
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            results.append(
                BenchmarkResult(
                    num_envs=num_envs,
                    step_latency_s=None,
                    env_fps=None,
                    peak_vram_gb=None,
                    error=str(e)[:200],
                )
            )
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            torch.cuda.empty_cache()
    return results


def print_benchmark_table(
    results: Sequence[BenchmarkResult],
    *,
    title: str | None = None,
) -> None:
    """Print a benchmark sweep as a fixed-width table."""
    width = 72
    if title:
        print("=" * width)
        print(title)
        print("=" * width)
    print(
        f"{'envs':>6} | {'step latency':>13} | {'env-FPS':>10} | {'peak VRAM':>10}"
    )
    print("-" * width)
    for r in results:
        if r.step_latency_s is None:
            print(f"{r.num_envs:>6} | ERR: {(r.error or '')[:50]}")
        else:
            assert r.env_fps is not None and r.peak_vram_gb is not None
            print(
                f"{r.num_envs:>6} | "
                f"{r.step_latency_s * 1000:>10.2f} ms | "
                f"{r.env_fps:>10,.0f} | "
                f"{r.peak_vram_gb:>7.2f} GB"
            )
