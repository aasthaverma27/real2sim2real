# RoboSnap fork notes

Vendored from https://github.com/Renforce-Dynamics/MuGS at
`df1203176cf34c55aa2bb7269411f83484ce3905` (2026-05-14), Apache-2.0.

All changes are confined to `src/mugs/sensors/gaussian_sensor.py` and are marked
with a `RoboSnap` comment. Nothing else upstream is touched, so the diff stays
small enough to rebase or upstream. Verified against RoboSnap's own renderer
(`scripts/mujoco_splat_render.py`) on `outputs/video_scene` from the capture
viewpoint at 720x1280: mean |diff| 0.76/255, 0.52% of pixels differing by more
than 16 - the residual is that our renderer forces square pixels while the
bridge passes the capture's real fx and fy (1062.3 vs 1066.5).

## Bug fixes (upstreamable)

1. **`cam_fovy` is degrees, not radians.** `_extract_camera_params` did
   `fx = (width/2) / tan(model.cam_fovy[id] / 2)` with a comment claiming the
   field is radians "affected by `<compiler angle="radian"/>`". MuJoCo always
   reports `cam_fovy` in degrees. For a normal 45-60 degree camera this computes
   `tan(22..30 radians)`, i.e. an arbitrary focal length, so the background is
   rendered at a completely wrong zoom. Upstream's examples hide this by passing
   `--gs-fx/--gs-fy` explicitly. Measured on our capture camera (fovy 61.935
   degrees, 854 px tall): correct `fy` = 711.6 px, upstream `fy` = **-887.9 px**
   - a negative focal length.

2. **The MuJoCo -> gsplat basis flip was incomplete.** `_render_3dgs_background`
   negated only column 2 of the camera rotation. A single column negation gives
   `det(R) = -1`, which is a mirror, not a rotation: MuJoCo cameras are x right,
   y up, z backward, and gsplat/OpenCV want x right, y down, z forward, so **y
   and z both flip**. Left as-is, the splat comes out vertically mirrored
   relative to the MuJoCo layer it is composited with. Measured on the same
   camera: `det(R)` = +1, `det(upstream z-only flip)` = **-1**,
   `det(y and z flip)` = +1. Configurable via `camera_convention` ("mujoco"
   flips both, "opencv" passes through).

3. **Non-square pixels from a vertical FOV.** `fx` was derived from the image
   width and `fy` from the height using the same vertical FOV, which stretches
   the background on any non-square image. Both now derive from the height.

## Behaviour added

4. **Full spherical harmonics** (`use_sh`, default on). Upstream keeps only the
   SH DC term, discarding view-dependent shading. Note this is a no-op for
   RoboSnap's current Lyra output, which is DC-only (`gravity_aligned_background
   *.ply` carries no `f_rest_*`); it matters for external splats - COLMAP or
   Nerfstudio trained, GS-Playground, DISCOVERSE - which do carry higher bands.

5. **Depth-aware compositing** (`depth_composite`). Upstream composites with a
   binary mask, so foreground geometry always draws over the splat even when it
   is physically behind it. The splat is now rendered `RGB+ED` and MuJoCo's
   depth buffer is read, so the mask is intersected with a z-test (with a
   `depth_tolerance` and a fall back to the plain mask if the test rejects most
   of the foreground, which would indicate a bad alignment rather than
   occlusion).

   **It is weaker than it sounds, and measurement says so.** gsplat's `ED` is the
   alpha-weighted *expected* depth along the ray, not the depth of the first
   surface. On a soft reconstruction (median gaussian opacity 0.05, heavily
   overlapping) the expected depth lands well beyond the visible surface: over
   the foreground mask of `outputs/video_scene` from the capture viewpoint, the
   splat's ED sits a median of **1.29 m** behind the mesh, and 0% of the mask is
   within 10 cm of it. So the z-test almost never fires - it is a guard against
   gross misalignment, not a mechanism for fine occlusion. Real occlusion between
   robot and room needs a renderer that resolves both in one pass; see
   `docs/batched_photoreal_env.md`.

6. **Renderer reuse and scene options.** `_render_mujoco_foreground` built and
   closed a `mujoco.Renderer` - a GL context - on *every* call, which is
   unusable in a stepping loop. It is now created once per (model, resolution)
   and released by `close()`. `disable_rangefinder` (default on) hides
   rangefinder rays, without which the G1's 512 lidar rays swamp the frame.

7. **Binary composite via select.** `np.where` instead of float alpha blending;
   the mask is binary, and at 720x1280 the float path cost more than the splat
   render itself (49.6 -> 41.7 ms per frame).

## The mjlab path (mugs_mjlab) needed three more fixes

`mugs_mjlab` was written against an older mjlab and does not run against mjlab
1.6.0 as shipped. Installed in its own env (`robosnap-mjlab`, mujoco 3.11.0 +
mujoco_warp 3.11.0) because mjlab pins those and installing it into
`robosnap-sim` would downgrade the working 3.12 stack.

8. **`TorchArray` drift.** `_get_camera_poses_batch` calls
   `wp.to_torch(self._mjwarp_data.cam_xpos)`. Current mjlab wraps its buffers in
   `mjlab.sim.sim_data.TorchArray`, a proxy sharing memory with the warp array
   and exposing it as `.wp_array`; passing the proxy to `wp.to_torch` raises
   `AttributeError: 'torch.device' object has no attribute 'is_cpu'`. Fixed by
   unwrapping `.wp_array` (and passing a `torch.Tensor` straight through).

9. **Segmentation is (object_id, object_type) pairs.**
   `_create_robot_masks_batch` treats the buffer as a single channel of geom ids,
   so `masks[seg == geom_id]` also matches the *type* channel, and the resulting
   `(N, H, W, 2)` mask then fails `expand(-1, -1, -1, 3)` in `_composite_batch`.
   Fixed by masking on `type == ObjType.GEOM` intersected with the wanted ids.

10. **The mask configuration cannot work on mjlab's own robot.**
    `robot_geom_names` defaults to `base_link, shoulder_link, arm_link,
    forearm_link, palm, left_finger_link, right_finger_link`; mjlab's YAM scene
    names its geoms `robot/base_collision`, `robot/link2_1_collision`, ... - **0
    of 7 resolve**, so the foreground is never composited. Passing the real names
    does not help either: mjlab names only *collision* geoms, its visual geoms
    are unnamed, and only visual geoms are rendered - mask coverage 0.0%.
    Selecting structurally (any rendered geom is foreground) works, but then
    `enabled_geom_groups=(0, 1, 2)` includes group 0, mjlab's terrain plane,
    which fills the frame: coverage 100%, splat entirely hidden. Restricting to
    `(2,)` gives a real 2.7% foreground.

    Note this is not a small bug: a name-based foreground list cannot express
    "the robot" in a scene whose visual geoms are unnamed.

11. **Their benchmark does not exercise the sensor.** `benchmark_mjlab_envs.py`
    reported 258 / 3,491 / 8,994 env-FPS at 1 / 16 / 64 envs on this machine
    *before* fixes 8-10 were applied. Touching `sensor.data` crashes without
    them, so those figures measure mjlab physics and mjlab's own renderer, not
    the Gaussian sensor.

12. **A reconstructed splat cannot be dropped into their task.** With the sensor
    finally rendering, the composite is black: mjlab's wrist camera sits at
    (0.176, -0.002, 0.254) while our splat's dense core spans y[0.48, 5.83],
    z[-1.91, -0.05]. Their scenes ship a PLY pre-aligned to the robot scene;
    a RoboSnap splat is aligned to the room it was captured in. Using one there
    means placing the robot in the reconstruction's frame, not the reverse.

13. **The mjlab sensor never converted the camera basis.** `_get_camera_poses_batch`
    built the camera-to-world pose straight from MuJoCo's `cam_xmat` and
    `_render_3dgs_batch_optimized` inverted it for gsplat. MuJoCo cameras look
    down -Z with +Y up; gsplat wants +Z forward, +Y down. With neither y nor z
    flipped the camera faces 180 degrees away from the scene, so every gaussian
    is behind it and the background renders **entirely black** - independent of
    how the splat is aligned. This is the same class of error as fix 2 in the
    standalone sensor, except there one column was flipped and here none were.
    After the fix, `robot/front_cam` renders a background covering 71.3% of the
    frame at mean 67.3/255.

14. **Background fovy was hardcoded.** `fov_y_deg = self.cfg.fovy or 45.0`, so
    unless the caller set it explicitly the background was rendered at 45 degrees
    regardless of the camera. mjlab's `robot/camera_d405` - the wrist camera
    `make_yam_lift_cube_gs_env_cfg` defaults to - is 58.0 degrees, so the
    background came out at the wrong zoom by default. Now falls back to the MJCF
    camera's own `cam_fovy`.
