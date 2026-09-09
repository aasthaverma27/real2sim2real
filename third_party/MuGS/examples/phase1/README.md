# Phase 1: Hybrid Rendering Test

**Date**: 2026-05-02  
**Status**: ✅ Complete

## Overview

Phase 1 验证混合渲染策略的可行性：

1. **MuJoCo** 渲染机器人本体 + 分割mask
2. **3DGS** (占位符) 渲染环境与物体
3. **Mask合成** 将两者合成为最终图像
4. **保存所有中间产物** 便于调试与分析

## 测试场景

- **机器人**: 5指夹爪 (palm + 5 fingers)
- **环境**: 桌子 (table) + 地面 (ground)
- **物体**: 杯子 (mug) + 盘子 (plate)
- **相机**: 2个视角 (main_cam, front_cam)

### Scene Hierarchy

```
World
├── Ground (plane with grid texture)
├── Table (box + 4 legs)
├── Robot Base
│   ├── Palm (box)
│   ├── Finger 1 (thumb, capsule)
│   ├── Finger 2 (capsule)
│   ├── Finger 3 (capsule)
│   ├── Finger 4 (capsule)
│   └── Finger 5 (capsule)
├── Mug (cylinder + handle)
└── Plate (flat cylinder)
```

## 渲染流程 (Hybrid Pipeline)

### Step 1: MuJoCo RGB

渲染完整场景（机器人 + 环境 + 物体）

**输出**: `*_1_mujoco_rgb.png`
- 分辨率: 640×480 (高分辨率) / 160×120 (低分辨率)
- 内容: 所有物体的简单着色渲染
- 用途: 提供机器人本体的视觉信息

### Step 2: Segmentation Mask

渲染分割ID图

**输出**: `*_2_segmentation.png`
- 每个像素对应一个geom ID
- 颜色映射使用 `tab20` colormap
- Unique IDs: 8-14 (取决于视角)

**Geom分类**:
- 机器人部件: palm, finger1~5_link (6个geoms)
- 环境: ground, table_top, table_leg1~4 (6个geoms)
- 物体: mug_body, mug_handle, plate_geom (3个geoms)

### Step 3: Robot Mask

提取机器人二值mask

**输出**: `*_3_robot_mask.png`
- 二值图像: 1=机器人, 0=环境/物体
- 机器人像素占比: 0.0~0.1% (取决于视角)
- 用途: mask合成的关键

**实现**:
```python
robot_geom_names = ["palm", "finger1_link", ..., "finger5_link"]
mask[seg_ids in robot_geom_ids] = 1
```

### Step 4: 3DGS Render (Placeholder)

占位符渲染（Phase 2将替换为真实gsplat）

**输出**: `*_4_3dgs_placeholder.png`
- 梯度背景 (蓝天 → 白色)
- 简单圆形模拟物体 (杯子=蓝色圆, 盘子=白色椭圆)
- **TODO**: 替换为 `gsplat.rasterize(...)`

**未来实现**:
```python
# Phase 2+: Real 3DGS rendering
from gsplat import rasterize
rendered = rasterize(
    means=gaussian_means,
    quats=gaussian_quats,
    scales=gaussian_scales,
    opacities=gaussian_opacities,
    colors=gaussian_colors,
    viewmats=camera_viewmat,
    Ks=camera_intrinsics,
    width=width, height=height
)
```

### Step 5: Composite

使用mask合成最终图像

**输出**: `*_5_composite.png`
- 公式: `composite = mask ? mujoco_rgb : gs_rgb`
- 机器人区域: 来自MuJoCo
- 环境/物体区域: 来自3DGS

**实现**:
```python
mask_3ch = np.stack([robot_mask] * 3, axis=-1)
composite = np.where(mask_3ch, mujoco_rgb, gs_rgb)
```

### Pipeline Comparison

**输出**: `*_pipeline_comparison.png`

6-panel可视化:
1. MuJoCo RGB
2. Segmentation IDs
3. Robot Mask
4. 3DGS Render
5. Composite Result
6. Difference Map (|MuJoCo - Composite|)

## 性能测试

### 高分辨率 (640×480)

- **MuJoCo RGB**: ~30-40 ms
- **Segmentation**: ~30-40 ms
- **Mask extraction**: <1 ms
- **3DGS placeholder**: <1 ms
- **Composite**: <1 ms
- **Total**: ~70-80 ms/frame

### 低分辨率 (160×120)

- **Total**: 66.19 ± 2.12 ms/frame
- **Throughput**: **15.1 FPS**
- **MuJoCo overhead主导** (分割渲染较慢)

**目标 vs 实际**:
- 目标 (3DGS阶段): 5000-8000 FPS
- 当前 (含MuJoCo): 15 FPS
- **瓶颈**: MuJoCo segmentation渲染
- **解决方案**: 
  - Phase 2: 用gsplat替换占位符 (预期 >5000 FPS)
  - 优化MuJoCo渲染设置 (考虑降低质量)

## 生成的文件

每个视角生成6个文件:

### main_cam (主视角)
- `main_cam_1_mujoco_rgb.png` - MuJoCo完整渲染
- `main_cam_2_segmentation.png` - 分割ID图
- `main_cam_3_robot_mask.png` - 机器人二值mask
- `main_cam_4_3dgs_placeholder.png` - 3DGS占位符
- `main_cam_5_composite.png` - 合成结果
- `main_cam_pipeline_comparison.png` - 流程对比图

### front_cam (前视角)
- `front_cam_*.png` - 同上

### lowres (低分辨率160×120)
- `lowres_*.png` - 用于性能测试

**总计**: 18个PNG文件

## 运行测试

```bash
cd /home/ununtu/metabot-workspace/mugs
python examples/phase1/hybrid_rendering_test.py
```

**输出目录**: `outputs/phase1_hybrid_test/`

## 关键发现

### ✅ 成功验证

1. **混合渲染可行**: Mask合成工作正常
2. **分割准确**: Robot mask正确提取机器人部件
3. **多视角支持**: 两个相机都能正常渲染
4. **中间产物完整**: 所有5步都有输出图像

### ⚠️ 待优化

1. **性能**: 当前15 FPS远低于5000 FPS目标
   - **原因**: MuJoCo渲染overhead
   - **计划**: Phase 2用gsplat替换，预期大幅提升
2. **3DGS质量**: 当前是占位符，不够真实
   - **计划**: Phase 2集成真实3DGS资产
3. **机器人可见性**: 某些视角机器人占比极小 (0.0%)
   - **计划**: 调整相机位置或机器人姿态

## Next Steps (Phase 2)

### 1. 集成真实3DGS渲染

```python
# Replace placeholder with gsplat
def render_3dgs_real(gaussian_params, camera_pose, width, height):
    from gsplat import rasterize
    rendered = rasterize(
        means=gaussian_params['means'],
        quats=gaussian_params['quats'],
        scales=gaussian_params['scales'],
        opacities=gaussian_params['opacities'],
        colors=gaussian_params['colors'],
        viewmats=camera_pose,
        Ks=intrinsics,
        width=width, height=height
    )
    return rendered
```

### 2. 下载真实3DGS资产

```bash
python scripts/data_collection/download_assets.py --preset quick
```

### 3. 实现GaussianSensor类

```python
# src/mugs/sensors/gaussian_sensor.py
class GaussianSensor:
    def __init__(self, config: GaussianSensorCfg):
        self.gaussians = {}  # object_id -> GaussianParams
        
    def load_object(self, object_id: str, ply_path: Path):
        """Load 3DGS PLY file"""
        
    def render(self, object_poses: dict, camera_pose: np.ndarray):
        """Render scene from camera"""
```

### 4. 性能基准测试

- 目标: 4096 parallel cameras @ 160×120
- 测试gsplat batched rendering
- 对比不同分辨率的FPS

## 参考资料

- **技术设计**: `docs/design/TECHNICAL_DESIGN.md`
- **混合渲染策略**: `HYBRID_RENDERING_STRATEGY.md`
- **TODO计划**: `TODO.md` Phase 1-2
- **项目概览**: `PROJECT_OVERVIEW.md`

---

**测试完成时间**: 2026-05-02  
**下一步**: Phase 2 - 集成gsplat与真实3DGS资产
