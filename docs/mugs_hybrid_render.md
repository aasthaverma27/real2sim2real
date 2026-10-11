# MuGS rendering of RoboSnap scenes

[MuGS](https://github.com/Renforce-Dynamics/MuGS) (MuJoCo Gaussian Splatting)
composites a 3D Gaussian-splat background behind MuJoCo's own render, using
MuJoCo's segmentation buffer to decide which pixels belong to the foreground.
It is vendored at `third_party/MuGS` (Apache-2.0) with a small set of fixes
recorded in [FORK_NOTES.md](../third_party/MuGS/FORK_NOTES.md).

`scripts/mugs_render.py` bridges MuGS's `GaussianSensor` to a RoboSnap scene, so
anything built on that sensor - rollout loops, and eventually mjlab batching -
runs on our reconstructions.

    bash scripts/run_mugs_render.sh --scene-dir outputs/myroom --capture-camera

## What the bridge has to reconcile

| | MuGS expects | RoboSnap has |
|---|---|---|
| frame | a COLMAP scene plus a `world_T_gs` transform | `gravity_aligned_background*.ply`, already in the MuJoCo world frame (`gravity_align_scene` bakes `transform_gravity_from_background` into the PLY), so no transform is needed |
| intrinsics | `fovy` off the MJCF camera | real calibration in `depth/camera.json`, passed explicitly; the MJCF `fovy` is only a fallback |
| foreground | a list of robot geom *names* | geoms named after whatever the VLM called each object, so the foreground is selected structurally: every visual geom that is not the floor, not a worldbody geom, and not collision-only (group 3) |

## Usage

    # single frame from the original capture viewpoint
    bash scripts/run_mugs_render.sh --scene-dir outputs/myroom --capture-camera

    # from an MJCF camera, at a different resolution
    bash scripts/run_mugs_render.sh --scene-dir outputs/myroom \
        --camera scene_view --width 1280 --height 720

    # layer inspection
    bash scripts/run_mugs_render.sh --scene-dir outputs/myroom --mode 3dgs_only
    bash scripts/run_mugs_render.sh --scene-dir outputs/myroom --mode mujoco_only

    # a rollout: step physics and render every frame through the sensor
    bash scripts/run_mugs_render.sh --scene-dir outputs/myroom --capture-camera \
        --hold --video outputs/myroom/mugs_rollout.mp4 --frames 120

    # agreement with RoboSnap's own renderer
    bash scripts/run_mugs_render.sh --scene-dir outputs/myroom --capture-camera \
        --no-depth-composite --compare outputs/myroom/ref_hybrid.png

`--no-depth-composite` reverts to upstream's binary mask, `--no-sh` to upstream's
DC-only colour. Both exist to reproduce upstream behaviour, not because they look
better.

The depth test is a weak guarantee, and the render reports why: gsplat's `ED` is
the alpha-weighted expected depth, not the first surface, so on a soft
reconstruction it sits a median of 1.29 m behind the mesh and the test almost
never fires. Treat it as a misalignment guard. For true occlusion between the
robot and the room, a renderer has to resolve both in one depth pass.

## Where it is used

`scripts/mugs_viewer.py` (the live hybrid window `start.sh` opens),
`scripts/elp_camera_publisher.py` and `scripts/realsim/vla_camera_server.py`
(the robot cameras) all render through this bridge. `scripts/mugs_render.py`
renders a scene offline: `bash scripts/run_mugs_render.sh --scene-dir outputs/myroom`.

## Measured performance

RTX 5090, `outputs/myroom` (2,575,456 gaussians, DC-only SH; MJCF with the
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
