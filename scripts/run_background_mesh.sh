#!/usr/bin/env bash
# Build a textured background MESH from the COLMAP reconstruction, the way
# Re3Sim does (scene_dense_mesh_refine_texture.usd). A splat only exists where a
# camera looked, so a robot eye 20 cm above the table renders black. A mesh is
# view-independent: once the tabletop and the room are surfaces, every viewpoint
# has something to show, and the splat becomes a colour overlay on top of it
# rather than the only source of pixels.
#
# OpenMVS resolves the image paths stored in the .mvs relative to the .mvs file
# itself, so every stage runs from inside the undistorted workspace.
set -eo pipefail
export PATH=${RS_OPENMVS_BIN%/OpenMVS}:${RS_OPENMVS_BIN:-/mnt/ssd2/Re3Sim/third_party/openmvs-install/bin/OpenMVS}:$PATH

SRC=${1:?usage: run_background_mesh.sh <colmap-workspace> [out-dir]}
OUT=${2:-$(dirname "$SRC")/roommesh}
RES=${RES:-2}
mkdir -p "$OUT"

if [ -d "$OUT/dense/images" ]; then
    echo "[mesh] 1/5 reusing the undistorted workspace in $OUT/dense"
else
    echo "[mesh] 1/5 undistorting $(ls "$SRC/images" | wc -l) images"
    colmap image_undistorter --image_path "$SRC/images" \
        --input_path "$SRC/colmap/sparse/0" --output_path "$OUT/dense" \
        --output_type COLMAP --max_image_size 1600
fi

cd "$OUT"
[ -f scene.mvs ] || { echo "[mesh] 2/5 InterfaceCOLMAP"
    InterfaceCOLMAP -i dense -o scene.mvs; }

echo "[mesh] 3/5 DensifyPointCloud (CPU, resolution-level $RES)"
DensifyPointCloud scene.mvs --resolution-level "$RES" --number-views 6 \
    --max-threads "$(nproc)"

echo "[mesh] 4/5 ReconstructMesh"
ReconstructMesh scene_dense.mvs --smooth 2

echo "[mesh] 5/5 TextureMesh"
TextureMesh scene_dense_mesh.mvs --export-type obj --resolution-level 1

echo "[mesh] done"; ls -la *.obj *.ply 2>/dev/null
