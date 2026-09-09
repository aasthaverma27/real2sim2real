#!/usr/bin/env python3
"""Env-parallel gsplat benchmark — push to 4096 environments."""
import time
import torch
from gsplat import rasterization

def make_scene(N, device='cuda'):
    m = torch.randn(N, 3, device=device) * 0.5
    q = torch.randn(N, 4, device=device); q = q / q.norm(dim=-1, keepdim=True)
    s = torch.ones(N, 3, device=device) * 0.01
    o = torch.ones(N, device=device) * 0.9
    c = torch.rand(N, 3, device=device)
    return m, q, s, o, c

def make_distinct_cameras(B, W, H, device='cuda'):
    torch.manual_seed(0)
    theta = torch.rand(B, device=device) * 2 * 3.14159
    phi = torch.rand(B, device=device) * 3.14159
    r = 2.0 + 0.5 * torch.rand(B, device=device)
    eye = torch.stack([r*torch.sin(phi)*torch.cos(theta),
                       r*torch.sin(phi)*torch.sin(theta),
                       r*torch.cos(phi)], dim=-1)
    fwd = -eye / eye.norm(dim=-1, keepdim=True)
    up = torch.tensor([0., 0., 1.], device=device).expand(B, -1)
    right = torch.cross(fwd, up, dim=-1)
    right = right / right.norm(dim=-1, keepdim=True)
    up2 = torch.cross(right, fwd, dim=-1)
    R = torch.stack([right, up2, -fwd], dim=-2)
    vm = torch.eye(4, device=device).expand(B, -1, -1).clone()
    vm[:, :3, :3] = R
    vm[:, :3, 3] = -torch.bmm(R, eye.unsqueeze(-1)).squeeze(-1)
    K = torch.tensor([[500., 0, W/2], [0, 500., H/2], [0, 0, 1.]], device=device)
    K = K.expand(B, -1, -1).contiguous()
    return vm.contiguous(), K

def bench(N, B, W, H, n_iters=10, warmup=3):
    torch.cuda.empty_cache()
    m, q, s, o, c = make_scene(N)
    vm, K = make_distinct_cameras(B, W, H)
    try:
        for _ in range(warmup):
            img, _, _ = rasterization(m, q, s, o, c, vm, K, W, H, packed=False)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iters):
            img, _, _ = rasterization(m, q, s, o, c, vm, K, W, H, packed=False)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / n_iters
        pm = torch.cuda.max_memory_allocated() / (1024**3)
        torch.cuda.reset_peak_memory_stats()
        return dt, B / dt, pm
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None, None

free_b, total_b = torch.cuda.mem_get_info()
print(f"Device: {torch.cuda.get_device_name(0)}  free={free_b/1e9:.1f}/{total_b/1e9:.1f} GB")
print()

for N, label in [(6_180,   "Kitchen-scale 6k gaussians"),
                 (50_000,  "Medium scene 50k gaussians"),
                 (100_000, "Real scene 100k gaussians"),
                 (500_000, "Large scene 500k gaussians")]:
    print("=" * 78)
    print(f"{label} @ 160×120")
    print("=" * 78)
    print(f"{'envs':>6} | {'latency':>10} | {'env-fps':>11} | {'per-env (µs)':>13} | {'peak mem':>10}")
    print("-" * 78)
    for B in [1, 16, 64, 256, 1024, 2048, 4096]:
        n_iters = max(3, 100 // max(B, 1))
        dt, fps, pm = bench(N, B, 160, 120, n_iters=n_iters)
        if dt is None:
            print(f"{B:>6} | OOM")
        else:
            print(f"{B:>6} | {dt*1000:>8.2f} ms | {fps:>11,.0f} | {dt*1e6/B:>11.2f} | {pm:>7.2f} GB")
    print()

# 4096 envs sweep across resolutions for kitchen scale
print("=" * 78)
print("4096 envs × resolution sweep (6k gaussians)")
print("=" * 78)
print(f"{'resolution':>12} | {'latency':>10} | {'env-fps':>11} | {'peak mem':>10}")
print("-" * 78)
for W, H in [(80, 60), (160, 120), (224, 224), (320, 240)]:
    dt, fps, pm = bench(6_180, 4096, W, H, n_iters=5)
    if dt is None:
        print(f"  {W}×{H}: OOM")
    else:
        print(f"  {W:>4}×{H:<4} | {dt*1000:>8.2f} ms | {fps:>11,.0f} | {pm:>7.2f} GB")
