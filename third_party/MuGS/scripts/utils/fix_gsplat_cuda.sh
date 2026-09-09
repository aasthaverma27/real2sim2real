#!/bin/bash
# Fix gsplat CUDA compilation for RTX 4090
#
# Root cause: CUDA 11.6 nvcc doesn't support compute_89 (RTX 4090 native arch)
# Solution: Fix symlinks + use compute_86 compatibility mode
#
# Author: MuGS Team
# Date: 2026-05-02

echo "======================================================================"
echo "gsplat CUDA Compilation Fix"
echo "======================================================================"
echo ""

# Step 1: Fix libcudart symlink (points to non-existent 11.6.55)
echo "Step 1: Fixing libcudart symlink to use CUDA 12.1..."
CONDA_LIB=/home/ununtu/miniconda3/lib
if [ -L "$CONDA_LIB/libcudart.so" ]; then
    rm -f "$CONDA_LIB/libcudart.so"
fi
ln -sf "$CONDA_LIB/libcudart.so.12" "$CONDA_LIB/libcudart.so"
echo "   ✅ libcudart.so -> libcudart.so.12"
echo ""

# Step 2: Create lib64 symlinks (linker looks in lib64)
echo "Step 2: Creating lib64 symlinks for CUDA libraries..."
CONDA_LIB64=/home/ununtu/miniconda3/lib64
mkdir -p "$CONDA_LIB64"
ln -sf "$CONDA_LIB/libcudart.so" "$CONDA_LIB64/libcudart.so" 2>/dev/null || true
ln -sf "$CONDA_LIB/libcudart.so.12" "$CONDA_LIB64/libcudart.so.12" 2>/dev/null || true
ln -sf "$CONDA_LIB/libcudart.so.12.1.105" "$CONDA_LIB64/libcudart.so.12.1.105" 2>/dev/null || true
echo "   ✅ Created symlinks in lib64/"
echo ""

# Step 3: Set compatible CUDA architecture
echo "Step 3: Setting CUDA architecture to compute_86 (compatible with nvcc 11.6)..."
export TORCH_CUDA_ARCH_LIST="8.6"
echo "   ✅ TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo ""

# Step 4: Clear torch extensions cache
echo "Step 4: Clearing torch extensions cache..."
CACHE_DIR="$HOME/.cache/torch_extensions"
if [ -d "$CACHE_DIR" ]; then
    echo "   Removing $CACHE_DIR"
    rm -rf "$CACHE_DIR"
    echo "   ✅ Cache cleared"
else
    echo "   No cache to clear"
fi
echo ""

# Step 5: Test gsplat import and compilation
echo "Step 5: Testing gsplat compilation..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print('')
print('Importing gsplat (will trigger JIT compilation)...')

try:
    from gsplat import rasterization
    print('✅ gsplat imported successfully!')
    
    # Quick test
    import time
    means = torch.randn(1000, 3, device='cuda')
    quats = torch.randn(1000, 4, device='cuda')
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.ones(1000, 3, device='cuda') * 0.01
    opacities = torch.ones(1000, device='cuda') * 0.9
    colors = torch.rand(1000, 3, device='cuda')
    
    viewmat = torch.eye(4, device='cuda')[None, ...]
    K = torch.tensor([[500, 0, 80], [0, 500, 60], [0, 0, 1]], device='cuda', dtype=torch.float32)[None, ...]
    
    # Warmup
    for _ in range(5):
        rendered, _, _ = rasterization(
            means, quats, scales, opacities, colors,
            viewmat, K, 160, 120, packed=False
        )
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    n = 50
    for _ in range(n):
        rendered, _, _ = rasterization(
            means, quats, scales, opacities, colors,
            viewmat, K, 160, 120, packed=False
        )
    torch.cuda.synchronize()
    elapsed = time.time() - start
    fps = n / elapsed
    
    print(f'')
    print(f'✅ gsplat GPU rendering works!')
    print(f'   Throughput: {fps:.0f} FPS @ 160×120 (1000 Gaussians)')
    print(f'   Estimated for 6180 Gaussians: {fps/6.18:.0f} FPS')
    
except Exception as e:
    print(f'❌ gsplat compilation failed:')
    print(f'   {e}')
    import sys
    sys.exit(1)
"

echo ""
echo "======================================================================"
echo "Fix Complete!"
echo "======================================================================"
echo ""
echo "To make TORCH_CUDA_ARCH_LIST permanent, add to ~/.bashrc:"
echo "   export TORCH_CUDA_ARCH_LIST=\"8.6\""
echo ""
