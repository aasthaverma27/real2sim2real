#!/bin/bash
# 快速展示 MuGS 测试结果
# Usage: ./scripts/show_results.sh

set -e

MUGS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MUGS_ROOT"

echo "════════════════════════════════════════════════════════════════════"
echo "MuGS 测试结果展示"
echo "════════════════════════════════════════════════════════════════════"
echo

# 检查输出目录
if [ ! -d "outputs" ]; then
    echo "❌ 未找到 outputs/ 目录"
    echo "   请先运行测试生成结果"
    exit 1
fi

# 统计输出
total_images=$(find outputs -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.gif" \) | wc -l)
total_dirs=$(find outputs -mindepth 1 -maxdepth 1 -type d | wc -l)

echo "📊 测试输出统计:"
echo "   - 输出目录: $total_dirs 个"
echo "   - 输出图像: $total_images 张"
echo

# 列出各阶段结果
echo "📁 各阶段结果目录:"
echo
ls -lh outputs/ | grep '^d' | awk '{print "   •", $NF}'
echo

# 关键图像
echo "🖼️  关键测试图像:"
echo

key_images=(
    "gaussian_sensor_demo/comparison.jpg:Phase 2.1 混合渲染对比"
    "gaussian_sensor_pretrained_demo/poses_grid_2x2.jpg:Phase 2.2 多姿态网格"
    "gaussian_sensor_visible_robot/poses_grid_3x2.jpg:Phase 3.1 机器人姿态"
    "gaussian_sensor_working_hybrid/poses_grid1_2x2.jpg:Phase 3.2 工作流 (前4步)"
    "gaussian_sensor_working_hybrid/poses_grid2_2x2.jpg:Phase 3.2 工作流 (后2步)"
    "hybrid_kitchen_sequence/robot_motion.gif:Phase 3.3 动画序列"
)

for entry in "${key_images[@]}"; do
    IFS=':' read -r path desc <<< "$entry"
    if [ -f "outputs/$path" ]; then
        size=$(du -h "outputs/$path" | cut -f1)
        echo "   ✅ $desc"
        echo "      outputs/$path ($size)"
    else
        echo "   ⚠️  $desc (未找到)"
    fi
done

echo
echo "════════════════════════════════════════════════════════════════════"
echo "查看方式"
echo "════════════════════════════════════════════════════════════════════"
echo
echo "1. 查看完整报告:"
echo "   cat docs/TEST_REPORT.md"
echo
echo "2. 查看测试规划:"
echo "   cat docs/testing_status_and_plan.md"
echo
echo "3. 浏览关键图像:"
echo "   cd docs/test_results && ls -lh"
echo
echo "4. 打开特定图像 (示例):"
echo "   xdg-open outputs/gaussian_sensor_demo/comparison.jpg"
echo
echo "5. 生成/更新报告:"
echo "   python scripts/generate_test_report.py"
echo
