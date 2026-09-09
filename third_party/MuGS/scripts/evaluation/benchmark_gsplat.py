#!/usr/bin/env python3
"""
Benchmark gsplat GPU Performance

Tests GPU rendering performance with gsplat library.
Target: >5000 FPS @ 160×120 resolution

Author: MuGS Team
Date: 2026-05-02
"""

import torch
import time
import numpy as np


def benchmark_gsplat_gpu():
    """Benchmark gsplat GPU rendering performance."""

    try:
        from gsplat import rasterization
    except ImportError:
        print("❌ gsplat not available")
        return False

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False

    print("="*70)
    print("gsplat GPU Performance Benchmark")
    print("="*70)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print()

    # Test parameters
    width, height = 160, 120
    num_gaussians = 10000
    num_iters = 100

    print(f"Test Configuration:")
    print(f"  Resolution: {width}×{height}")
    print(f"  Gaussians: {num_gaussians:,}")
    print(f"  Iterations: {num_iters}")
    print()

    # Create test data on GPU
    print("📊 Creating test data...")
    means = torch.randn(num_gaussians, 3, device='cuda') * 0.5

    # Normalized quaternions
    quats = torch.randn(num_gaussians, 4, device='cuda')
    quats = quats / quats.norm(dim=-1, keepdim=True)

    scales = torch.ones(num_gaussians, 3, device='cuda') * 0.01
    opacities = torch.ones(num_gaussians, device='cuda') * 0.9
    colors = torch.rand(num_gaussians, 3, device='cuda')

    # Camera parameters
    viewmat = torch.eye(4, device='cuda')[None, ...]  # (1, 4, 4)
    K = torch.tensor([
        [500, 0, width/2],
        [0, 500, height/2],
        [0, 0, 1]
    ], device='cuda')[None, ...]  # (1, 3, 3)

    # Warmup
    print("🔥 Warming up...")
    for _ in range(10):
        rendered, _, _ = rasterization(
            means, quats, scales, opacities, colors,
            viewmat, K, width, height, packed=False
        )

    torch.cuda.synchronize()
    print("✅ Warmup complete")
    print()

    # Benchmark
    print(f"⏱️  Running {num_iters} iterations...")
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(num_iters):
        rendered, _, _ = rasterization(
            means, quats, scales, opacities, colors,
            viewmat, K, width, height, packed=False
        )

    torch.cuda.synchronize()
    elapsed = time.time() - start

    # Results
    fps = num_iters / elapsed
    ms_per_frame = (elapsed / num_iters) * 1000

    print()
    print("="*70)
    print("Results")
    print("="*70)
    print(f"Total time: {elapsed:.2f} s")
    print(f"Time per frame: {ms_per_frame:.2f} ms")
    print(f"Throughput: {fps:.0f} FPS")
    print()
    print(f"Target: 5000+ FPS")
    if fps >= 5000:
        print(f"Status: ✅ PASS ({fps/5000:.1f}x target)")
    else:
        print(f"Status: ⚠️  BELOW TARGET ({fps/5000:.1%})")
    print()

    # Memory usage
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    print(f"GPU Memory:")
    print(f"  Allocated: {allocated:.1f} MB")
    print(f"  Reserved: {reserved:.1f} MB")
    print()

    return fps >= 5000


def benchmark_batched():
    """Benchmark batched rendering (multiple cameras)."""

    try:
        from gsplat import rasterization
    except ImportError:
        return

    print("="*70)
    print("Batched Rendering Benchmark (VLA Use Case)")
    print("="*70)
    print()

    batch_sizes = [1, 16, 64, 256, 1024, 4096]
    width, height = 160, 120
    num_gaussians = 10000

    # Create test data
    means = torch.randn(num_gaussians, 3, device='cuda') * 0.5
    quats = torch.randn(num_gaussians, 4, device='cuda')
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.ones(num_gaussians, 3, device='cuda') * 0.01
    opacities = torch.ones(num_gaussians, device='cuda') * 0.9
    colors = torch.rand(num_gaussians, 3, device='cuda')

    print(f"Batch Size | FPS/Camera | Total FPS | Latency")
    print("-" * 70)

    for batch_size in batch_sizes:
        # Create batched camera parameters
        viewmat = torch.eye(4, device='cuda')[None, ...].expand(batch_size, -1, -1)
        K = torch.tensor([
            [500, 0, width/2],
            [0, 500, height/2],
            [0, 0, 1]
        ], device='cuda')[None, ...].expand(batch_size, -1, -1)

        # Warmup
        for _ in range(5):
            rendered, _, _ = rasterization(
                means, quats, scales, opacities, colors,
                viewmat, K, width, height, packed=False
            )

        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        n_iters = max(1, 100 // batch_size)  # Fewer iters for large batches

        for _ in range(n_iters):
            rendered, _, _ = rasterization(
                means, quats, scales, opacities, colors,
                viewmat, K, width, height, packed=False
            )

        torch.cuda.synchronize()
        elapsed = time.time() - start

        total_renders = n_iters * batch_size
        fps_per_camera = total_renders / elapsed
        total_fps = batch_size * (n_iters / elapsed)
        latency_ms = (elapsed / n_iters) * 1000

        print(f"{batch_size:10d} | {fps_per_camera:10.0f} | {total_fps:9.0f} | {latency_ms:6.1f} ms")

    print()


if __name__ == "__main__":
    success = benchmark_gsplat_gpu()
    print()
    benchmark_batched()

    if success:
        print("\n✅ GPU rendering operational and performant")
        exit(0)
    else:
        print("\n⚠️  GPU rendering below performance target")
        exit(1)
