# G1 apple scene — the full command list

Every step below was run to build `outputs/g1_ok/`. Two interpreters are needed:

    PY_ASSET = /mnt/ssd2/conda/envs/robosnap-asset/bin/python   # pipeline stages
    PY_LYRA  = /mnt/ssd2/conda/envs/robosnap-lyra/bin/python    # Lyra, gsplat, COLMAP
    SIM      = /mnt/ssd2/conda/envs/robosnap-sim/bin/python     # mujoco, sapien, coacd

    set -a; . configs/auto_pipeline.env; set +a

`docs/real2sim_experiments.md` carries the findings behind these commands -
what was measured, which approaches failed and why. Read it before changing
the alignment or the background pipeline.

## 0. Choose the source video, and re-encode it upright

Phone video carries `rotation=-90` as METADATA. Lyra's readers ignore it, so DA3
and VIPE - both trained on upright images - receive sideways frames and
reconstruct a scene lying on its side. Re-encoding bakes the rotation into the
pixels; ffmpeg auto-rotates on input, so a plain re-encode is enough.

    ffmpeg -i "Unitree g1/20260903_161432.mp4" -an -c:v libx264 -crf 16 out.mp4

## 1. Trim so YOUR frame becomes frame 0

`robosnap/refinement/gravity_align_scene.py:551` does

    background_camera = pad_extrinsic(cameras["w2c_render"][0])
    status = "aligned_first_camera"

That single line is the only thing tying the objects to the background: it
assumes the background video's FIRST frame is the object image. Trimming makes
that true for whichever frame you want.

    ffmpeg -ss 17.5 -i "Unitree g1/20260903_161432.mp4" -an -c:v libx264 -crf 16 \
        inputs/g1_f_video.mp4
    ffmpeg -i inputs/g1_f_video.mp4 -vframes 1 inputs/g1_f_frame0.png

## 2. Objects: masks, SAM-3D meshes, depth, ICP, gravity, physics

    INPUT_IMAGE=inputs/g1_f_frame0.png \
    OUTPUT_DIR=outputs/g1_ok \
    OBJECT_PROMPTS="the round wooden table,the red apple,the woven brown basket" \
    SF_REAL2SIM_COLLISION_METHOD=coacd \
      bash scripts/run_object_pipeline.sh

The Lyra background stage inside this script OOMs and is expected to - the
script's own comment says so. Everything through physics refinement completes.

## 3. Background: Lyra on the SAME video

Run this only after step 2 finishes. SAM-3D peaks at ~17 GB and Lyra at ~18 GB;
together they exceed 31 GB and one of them dies.

    $PY_ASSET -m robosnap.background.run_lyra2_gs_recon \
      --input-video inputs/g1_f_video.mp4 \
      --output-dir outputs/g1_ok/background/gs \
      --lyra-dir third_party/lyra --python $PY_LYRA \
      --checkpoint-dir checkpoints/lyra --device cuda:0 \
      --max-frames 240 --da3-max-frames 16 --max-resolution 640 \
      --gs-down-ratio 1 --vipe --force --require-render

`--max-resolution` above 640 OOMs at 16 DA3 views; 768 and 896 both failed.

## 4. Physics properties and the MJCF

    $PY_ASSET scripts/estimate_properties.py --scene-dir outputs/g1_ok --mode lookup
    MUJOCO_GL=egl $SIM scripts/export_mujoco.py --scene-dir outputs/g1_ok --static table

## 5. Metric rescale

VGGT is scale-free (`depth/camera.json` says `facebook/VGGT-1B`), so the export
is correct in proportion and wrong in absolute size. One global factor, anchored
on the measured table height, and cross-checked against the measured basket:

    MUJOCO_GL=egl $SIM scripts/rescale_scene.py \
      --scene outputs/g1_ok/mujoco/scene.xml \
      --output outputs/g1_ok/mujoco/scene_scaled.xml \
      --table-body obj0_round_wooden_table --table-height 0.75 \
      --apple-body obj1_red_apple --basket-body obj2_woven_brown_basket \
      --check-basket 0.245 0.19

    -> scale x1.18372, basket predicted 0.234 x 0.181 vs measured 0.245 x 0.190
       (4.7% / 4.7%)

## 6. Gravity-align the background to the objects

    $PY_ALIGN -m robosnap.refinement.gravity_align_scene \
      --scene-dir outputs/g1_ok \
      --foreground-glb outputs/g1_ok/refined_scene.glb \
      --background-ply outputs/g1_ok/background/gs/reconstructed_scene.ply \
      --background-cameras outputs/g1_ok/background/gs/cameras.npz \
      --vggt-dir outputs/g1_ok/depth --plane-mask outputs/g1_ok/support_mask.png \
      --output-foreground /tmp/fg.glb \
      --output-background outputs/g1_ok/gravity_aligned_background.ply \
      --transform-json outputs/g1_ok/gravity_alignment_bg.json --seed 0

## 7. Move the splat into the rescaled scene

`transform_gravity_from_background` is RIGID - rotation and translation only. The
Lyra background keeps its own arbitrary scale, so it still needs the same global
factor the objects got, and then a similarity that lands its table on MuJoCo's.

    $PY_LYRA scripts/transform_splat.py \
      --ply outputs/g1_ok/gravity_aligned_background.ply \
      --transform T_gravity_to_scene.npy \
      --out outputs/g1_ok/bg_scene.ply

`transform_splat.py` moves means, multiplies scales, and composes orientations.
A version that moves only the means turns the render into streaks.

## 8. Add the robot

    MUJOCO_GL=egl $SIM scripts/add_robot.py \
      --scene-dir outputs/g1_ok \
      --scene-xml outputs/g1_ok/mujoco/scene_scaled.xml \
      --robot assets/robots/g1/g1_29dof_with_hand.xml \
      --at <x> <y> --yaw <rad> --strip-rangefinders \
      --output outputs/g1_ok/mujoco/scene_g1.xml

`--strip-rangefinders` removes 72 sensors measured at a 206x physics slowdown.

## 9. Render through MuGS, and view

    export PATH=/mnt/ssd2/conda/envs/robosnap-sim/bin:$PATH   # gsplat needs ninja
    MUJOCO_GL=egl $SIM scripts/mugs_render.py \
      --scene-dir outputs/g1_ok --mjcf outputs/g1_ok/mujoco/scene_g1.xml \
      --ply outputs/g1_ok/bg_aligned.ply \
      --gravity-json outputs/g1_ok/gravity_alignment_bg.json \
      --camera cap --mode hybrid --width 480 --height 854 --hold --pin-base g1_ \
      --output /tmp/hyb.png --vram-cap-gb 26

    DISPLAY=:1 MUJOCO_GL=glfw $SIM scripts/hybrid_viewer.py \
      --scene-dir outputs/g1_ok --mjcf outputs/g1_ok/mujoco/scene_g1.xml \
      --ply outputs/g1_ok/bg_aligned.ply \
      --look-at <x> <y> 0.750 --distance 1.2 --elevation -25 --azimuth <deg>

Watch `splat-behind-mesh gap` in the mugs_render output. Near +0.02 m means the
layers agree; metres means the alignment is wrong, not the reconstruction.

## Bugs fixed along the way (all pre-existing)

| file | bug |
|---|---|
| `media_utils.py` | `if img is None:` mis-indented - `.jpg` raised UnboundLocalError, single-image mode had never run |
| `media_utils.py` | `IMREAD_UNCHANGED` skips EXIF orientation - photos loaded sideways |
| `inference_interactive_videoseg.py` | single-image mode wrote `obj_dir/000000.png`, but `sam3d_bridge` globs `top*_mask/` - every object was silently skipped at 3D generation |
| `sam-3d-objects/run_inference.py` | passed `attention_logger` to `InferencePipelinePointMap.run`, which does not accept it - single-view inference could not complete |
| `sam-3d-objects/inference_pipeline.py` | boolean mask written into an 8-bit alpha channel gave 0/1 instead of 0/255; later divided by 255 - every `> 0.5` test failed and meshes were built at a silent fallback scale |
| `img_and_mask_transforms.py` | `.max()` on an empty tensor raised inside the guard meant to detect an empty mask |
| `mugs_render.py` | `--fovy` changed the reported value but not fx/fy - splat rendered at the wrong field of view (36% -> 7.5% black) |
| `hybrid_viewer.py` | orbit target hard-coded from an older scene, so the camera started underground |

SAM3's Triton connected-components kernel caps at ~4.19 MP
(`ceil(H/4)*ceil(W/16) <= 65535`). 12 MP stills crash it; resize to 1536x2048.
