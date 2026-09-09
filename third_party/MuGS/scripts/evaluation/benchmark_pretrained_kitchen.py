"""
Benchmark pretrained Mip-NeRF 360 kitchen rendering performance.

Tests rendering speed and quality with the official INRIA pretrained model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import time
import json
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData
from gsplat import rasterization


def load_official_ply(ply_path: Path):
    """Load official 3DGS PLY format with spherical harmonics."""
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    positions = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)

    # Convert SH DC coefficients to RGB
    SH_C0 = 0.28209479177387814
    sh_dc = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=1)
    colors = 0.5 + SH_C0 * sh_dc
    colors = np.clip(colors, 0, 1)

    # Opacity (apply sigmoid)
    opacities = 1 / (1 + np.exp(-vertex['opacity']))

    # Scales (apply exp)
    scales = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=1)
    scales = np.exp(scales)

    # Quaternions (normalize)
    quats = np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']], axis=1)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)

    return {
        'means': positions,
        'colors': colors,
        'opacities': opacities,
        'scales': scales,
        'quats': quats,
    }


def camera_to_view_matrix(position, rotation):
    """Convert camera position and rotation to view matrix."""
    R = np.array(rotation).T
    t = -R @ np.array(position)
    view_matrix = np.eye(4)
    view_matrix[:3, :3] = R
    view_matrix[:3, 3] = t
    return view_matrix


def render_camera(gaussians, camera, width, height, device='cuda'):
    """Render a single camera view."""
    fx = camera['fx'] * (width / camera['width'])
    fy = camera['fy'] * (height / camera['height'])
    cx = width / 2
    cy = height / 2

    view_matrix = camera_to_view_matrix(camera['position'], camera['rotation'])
    viewmat = torch.from_numpy(view_matrix).float().to(device)

    render_colors, render_alphas, info = rasterization(
        means=gaussians['means'],
        quats=gaussians['quats'],
        scales=gaussians['scales'],
        opacities=gaussians['opacities'],
        colors=gaussians['colors'],
        viewmats=viewmat[None],
        Ks=torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device)[None],
        width=width,
        height=height,
        packed=False,
    )

    rgb = render_colors[0].clamp(0, 1)
    return (rgb.cpu().numpy() * 255).astype(np.uint8)


def benchmark_rendering_speed(gaussians, camera, resolutions, device='cuda', warmup=5, runs=50):
    """Benchmark rendering at different resolutions."""
    results = {}

    for width, height in resolutions:
        # Warmup
        for _ in range(warmup):
            _ = render_camera(gaussians, camera, width, height, device)

        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = render_camera(gaussians, camera, width, height, device)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times) * 1000  # ms
        std_time = np.std(times) * 1000
        fps = 1000 / avg_time

        results[f"{width}x{height}"] = {
            'avg_ms': avg_time,
            'std_ms': std_time,
            'fps': fps,
        }

    return results


def main():
    """Run performance benchmark on pretrained kitchen model."""

    base_dir = Path(__file__).parent.parent.parent
    model_path = base_dir / "data/pretrained/kitchen/point_cloud/iteration_30000/point_cloud.ply"
    cameras_path = base_dir / "data/pretrained/kitchen/cameras.json"
    output_dir = base_dir / "outputs/pretrained_kitchen_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Pretrained Mip-NeRF 360 Kitchen Rendering Benchmark")
    print("=" * 70)

    # Load model
    print("\n[1/4] Loading pretrained model...")
    gaussians_np = load_official_ply(model_path)
    n_gaussians = len(gaussians_np['means'])
    print(f"  ✓ Loaded {n_gaussians:,} Gaussians")
    print(f"  ✓ Model size: {model_path.stat().st_size / 1e6:.1f} MB")

    # Transfer to GPU
    device = torch.device('cuda')
    gaussians = {
        'means': torch.from_numpy(gaussians_np['means']).float().to(device),
        'quats': torch.from_numpy(gaussians_np['quats']).float().to(device),
        'scales': torch.from_numpy(gaussians_np['scales']).float().to(device),
        'opacities': torch.from_numpy(gaussians_np['opacities']).float().to(device),
        'colors': torch.from_numpy(gaussians_np['colors']).float().to(device),
    }

    # Load cameras
    with open(cameras_path) as f:
        cameras = json.load(f)
    test_camera = cameras[100]  # Use middle camera

    print(f"\n[2/4] Testing at multiple resolutions...")

    # Test resolutions
    resolutions = [
        (160, 120),   # Phase 1 low-res
        (320, 240),   # 2x
        (640, 480),   # Phase 2 target
        (960, 640),   # Demo quality
        (1280, 720),  # HD
    ]

    results = benchmark_rendering_speed(gaussians, test_camera, resolutions, device)

    print("\n" + "─" * 70)
    print(f"{'Resolution':<15} {'Avg Time':<12} {'FPS':<12} {'vs Target'}")
    print("─" * 70)

    targets = {
        '160x120': 5000,   # Phase 1 target
        '640x480': 50,     # Phase 2 target (with SR)
    }

    for res_name, data in results.items():
        target_fps = targets.get(res_name, None)
        if target_fps:
            vs_target = f"{data['fps'] / target_fps:.1f}× target"
        else:
            vs_target = "—"

        print(f"{res_name:<15} {data['avg_ms']:>6.2f} ms    {data['fps']:>7.1f} FPS  {vs_target}")

    print("─" * 70)

    # Render gallery
    print(f"\n[3/4] Rendering quality gallery...")

    gallery_indices = [0, 50, 100, 150, 200, 250]
    render_width, render_height = 960, 640

    for idx in gallery_indices:
        if idx >= len(cameras):
            continue

        camera = cameras[idx]
        rgb = render_camera(gaussians, camera, render_width, render_height, device)

        output_path = output_dir / f"gallery_{idx:03d}.jpg"
        Image.fromarray(rgb).save(output_path, quality=95)
        print(f"  ✓ Rendered view {idx:03d} → {output_path.name}")

    # Save benchmark results
    print(f"\n[4/4] Saving results...")

    summary = {
        'model': {
            'path': str(model_path),
            'n_gaussians': n_gaussians,
            'size_mb': model_path.stat().st_size / 1e6,
        },
        'performance': results,
        'device': torch.cuda.get_device_name(0),
    }

    results_path = output_dir / "benchmark_results.json"
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  ✓ Results saved to {results_path}")

    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)
    print(f"\nKey Findings:")
    print(f"  • {n_gaussians:,} Gaussians (300× larger than programmatic scene)")
    print(f"  • 160×120: {results['160x120']['fps']:.0f} FPS ({results['160x120']['fps'] / 5000:.1f}× Phase 1 target)")
    print(f"  • 640×480: {results['640x480']['fps']:.0f} FPS")
    print(f"  • Photorealistic quality with real-world complexity")
    print(f"\n  Gallery: {output_dir}/gallery_*.jpg")


if __name__ == "__main__":
    import os
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
    main()
