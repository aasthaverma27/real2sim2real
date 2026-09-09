# MuGS rendering of RoboSnap scenes

[MuGS](https://github.com/Renforce-Dynamics/MuGS) (MuJoCo Gaussian Splatting)
composites a 3D Gaussian-splat background behind MuJoCo's own render, using
MuJoCo's segmentation buffer to decide which pixels belong to the foreground.
It is vendored at `third_party/MuGS` (Apache-2.0) with a small set of fixes
recorded in [FORK_NOTES.md](../third_party/MuGS/FORK_NOTES.md).

`scripts/mugs_render.py` bridges MuGS's `GaussianSensor` to a RoboSnap scene, so
anything built on that sensor - rollout loops, and eventually mjlab batching -
runs on our reconstructions.

    bash scripts/run_mugs_render.sh --scene-dir outputs/video_scene --capture-camera

## What the bridge has to reconcile

| | MuGS expects | RoboSnap has |
|---|---|---|
| frame | a COLMAP scene plus a `world_T_gs` transform | `gravity_aligned_background*.ply`, already in the MuJoCo world frame (`gravity_align_scene` bakes `transform_gravity_from_background` into the PLY), so no transform is needed |
| intrinsics | `fovy` off the MJCF camera | real calibration in `depth/camera.json`, passed explicitly; the MJCF `fovy` is only a fallback |
| foreground | a list of robot geom *names* | geoms named after whatever the VLM called each object, so the foreground is selected structurally: every visual geom that is not the floor, not a worldbody geom, and not collision-only (group 3) |

## Usage

    # single frame from the original capture viewpoint
    bash scripts/run_mugs_render.sh --scene-dir outputs/video_scene --capture-camera

    # from an MJCF camera, at a different resolution
    bash scripts/run_mugs_render.sh --scene-dir outputs/video_scene \
        --camera scene_view --width 1280 --height 720

    # layer inspection
    bash scripts/run_mugs_render.sh --scene-dir outputs/video_scene --mode 3dgs_only
    bash scripts/run_mugs_render.sh --scene-dir outputs/video_scene --mode mujoco_only

    # a rollout: step physics and render every frame through the sensor
    bash scripts/run_mugs_render.sh --scene-dir outputs/video_scene --capture-camera \
        --hold --video outputs/video_scene/mugs_rollout.mp4 --frames 120

    # agreement with RoboSnap's own renderer
    bash scripts/run_mugs_render.sh --scene-dir outputs/video_scene --capture-camera \
        --no-depth-composite --compare outputs/video_scene/ref_hybrid.png

`--no-depth-composite` reverts to upstream's binary mask, `--no-sh` to upstream's
DC-only colour. Both exist to reproduce upstream behaviour, not because they look
better.

The depth test is a weak guarantee, and the render reports why: gsplat's `ED` is
the alpha-weighted expected depth, not the first surface, so on a soft
reconstruction it sits a median of 1.29 m behind the mesh and the test almost
never fires. Treat it as a misalignment guard. For true occlusion between the
robot and the room, use the renderer that resolves both in one pass -
[batched_photoreal_env.md](batched_photoreal_env.md).

## Relationship to the existing scripts

`scripts/mujoco_splat_render.py` and `scripts/hybrid_viewer.py` implement the
same compositing idea directly and stay as they are:

* **`mujoco_splat_render.py`** - one frame, no dependency on MuGS. Still the
  reference: the bridge is validated against it.
* **`hybrid_viewer.py`** - interactive orbiting viewer with layer cycling.
* **`mugs_render.py`** - use when the sensor API matters: a stepping rollout, or
  anything that will later run under mjlab.

## Measured performance

RTX 5090, `outputs/video_scene` (2,575,456 gaussians, DC-only SH; MJCF with the
G1, a table and two objects, 260 geoms):

| resolution | splat | MuJoCo passes | frame (bg cached) | peak VRAM |
|---|---|---|---|---|
| 720x1280, depth composite | 5.9 ms | 22.9 ms | 41.7 ms (24 FPS) | 0.42 GiB |
| 720x1280, mask composite | 6.1 ms | 20.3 ms | 36.9 ms (27 FPS) | 0.42 GiB |
| 480x854, depth composite | 4.0 ms | 15.2 ms | 23.8 ms (42 FPS) | 0.39 GiB |

With a fixed camera the splat is rendered once and cached; only the MuJoCo layer
is redrawn per frame.

**Physics, not rendering, is the bottleneck for a rollout.** `mj_step` on this
scene costs 6.97 ms (143 steps/s, 0.29x realtime), so 30 Hz control means 17
steps and about 8 frames/s - below the renderer's 42 FPS at the same resolution.
Chasing render throughput before moving physics to the GPU would be optimising
the wrong half.

Peak VRAM stays under 0.5 GiB, three orders below the card, so the memory
envelope in MuGS's own docs (500k gaussians OOM-ing past ~256 envs) is about
per-view batched intermediates, not the gaussian store. `--vram-cap-gb` (default
31) sets a hard per-process ceiling and refuses to start if the GPU is already
over it.

## Splat distillation

`scripts/distill_splat.py` shrinks a background splat to a gaussian budget and
reports what it cost, so env count can be traded against fidelity with numbers
rather than guesses.

    bash scripts/run_distill_splat.sh --scene-dir outputs/video_scene \
        --sweep 1200000 600000 300000 150000 100000        # measure only
    bash scripts/run_distill_splat.sh --scene-dir outputs/video_scene \
        --target 100000 --finetune 6000        # writes <scene>/splats/*_distilled_100k.ply

Gaussians are ranked by how much they contribute to the *capture trajectory*:
for each of the 60 recorded cameras, a gaussian that lands inside the frame
scores its opacity times its projected area. Anything the sweep never saw is
dropped first, which suits a background splat whose valid view volume is only as
big as the sweep that produced it - here 0.22 m of camera travel.

**Pruning alone is not enough, and fine-tuning is not optional.** The survivors
still carry parameters fitted alongside 25x more neighbours, so everything the
removed gaussians contributed is simply missing:

| 100k gaussians (3.9% of 2.58M) | PSNR vs the full splat |
|---|---|
| pruned only | 16.87 dB |
| + 1500 iterations of self-distillation | 32.89 dB |
| + 6000 iterations | 35.10 dB |

Self-distillation needs no ground truth: the full splat is the teacher and its
own renders from the 50 fitting views are the targets, with 10 views held out for
the PSNR. Only the kept parameters move. Note the position learning rate must be
scaled by the **camera-trajectory** extent (0.22 m here), not the point cloud's
bounding box - this reconstruction spans 138 m of far-field junk around a 5 m
room, and using that scaled the position LR ~700x too high, drifting 60% of the
gaussians out of frame and capping recovery at 24.9 dB.

End to end, the distilled splat is a drop-in (`--ply`): at 720x1280 against the
full-splat render it differs by mean 1.47/255 with 1.76% of pixels off by more
than 16, at 27.4 FPS with **0.06 GiB** peak VRAM instead of 0.42 GiB, from a
6.5 MB file instead of 167 MB.

About 19% of the fine-tuned gaussians end below 0.01 opacity, so a second prune
pass would reach roughly 80k at the same quality. Not implemented.

## Next step: batched envs

Superseded - see [batched_photoreal_env.md](batched_photoreal_env.md). MuJoCo
Warp turned out to provide both a batched renderer and native splat primitives,
so `mugs_mjlab` is not needed: the batched env reaches 3,522 photoreal
frames/s at 256 worlds, with physics alone at 988,859 env-steps/s.
