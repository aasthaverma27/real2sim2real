"""Ready-to-use mjlab task configs + benchmark helpers for MuGS sensors.

Drop-in env factories that attach a `GaussianSensorMjlab` to mjlab tasks,
plus a small benchmark runner so anyone can reproduce the multi-env
throughput numbers in `README.md` against their own scenes/PLYs.

Example — build an env and benchmark it::

    from mugs_mjlab.tasks import (
        make_yam_lift_cube_gs_env_cfg,
        kitchen_ply_path,
        benchmark_env_cfg,
        print_benchmark_table,
    )

    def factory(num_envs):
        return make_yam_lift_cube_gs_env_cfg(
            ply_path=kitchen_ply_path(),
            num_envs=num_envs,
        )

    results = benchmark_env_cfg(factory, env_counts=(1, 64, 1024, 4096))
    print_benchmark_table(results)
"""

from mugs_mjlab.tasks.benchmark import (
    BenchmarkResult,
    benchmark_env_cfg,
    print_benchmark_table,
)
from mugs_mjlab.tasks.yam_lift_cube_gs import (
    kitchen_ply_path,
    make_yam_lift_cube_gs_env_cfg,
)

__all__ = [
    "BenchmarkResult",
    "benchmark_env_cfg",
    "print_benchmark_table",
    "kitchen_ply_path",
    "make_yam_lift_cube_gs_env_cfg",
]
