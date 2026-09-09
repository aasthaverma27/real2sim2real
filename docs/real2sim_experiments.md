# Real2sim experiments — G1 apple scene (`outputs/g1_ok`)

Findings log for the photoreal G1 pick-and-place scene: what was measured, what
worked, what did not, and why. `RUNBOOK_g1.md` has the command list; this file
has the reasoning, so a failed approach is not tried a second time.

Scene under test throughout: source video `Unitree g1/20260903_161432.mp4`,
trimmed at 17.5 s so the chosen frame becomes frame 0. Background splat
`outputs/g1_ok/bg_lm.ply` (10,282,496 Gaussians, Lyra/VIPE + DA3).

---

## F1 — MuGS and Re3Sim composite in opposite directions

Re3Sim (`re3sim/tasks/pick_and_place.py:480-523`):

```python
table_mask  = segment_data == name_to_indexes["table"]
ground_mask = segment_data == name_to_indexes["ground"]
mix_mask    = table_mask | ground_mask
mixed_image = np.where(mix_mask, gaussian_rendered_background, sim_only_image)
```

They render the **simulator**, segment it, and substitute the splat *only* on
table and ground pixels. No depth test. The mesh owns every silhouette; the
splat only supplies colour inside the mesh's own footprint, so a table cannot
appear twice however badly the splat is aligned.

MuGS does the reverse: splat everywhere the foreground mask (plus a depth test)
says "not robot". Any misalignment therefore shows as a doubled edge.

Implemented Re3Sim's rule in MuGS as `GaussianSensorConfig.splat_geom_ids`
(`third_party/MuGS/src/mugs/sensors/gaussian_sensor.py`), exposed as
`--splat-geoms` on `scripts/mugs_render.py` and `scripts/mugs_viewer.py`. Where
the splat has no coverage the mesh's own render is kept (`_last_bg_alpha > 0.5`),
so a novel viewpoint degrades to the mesh instead of to black.

Result from the capture camera: fully photoreal tabletop carrying the mesh's
silhouette, no double table, nothing pruned. From the robot's own camera it
falls apart — see F6.

## F2 — Re3Sim's background scan contains no manipulated objects

Their items are separate USD assets (`data/items/orange_bottle3.usd`) whose
position is randomised every episode (`ItemPoseRandom: item_random_range`). If
an item were baked into the splat, every episode would show a ghost at the scan
pose. `re3sim/scripts/reconstruct.py` (355 lines) contains no mask, no
segmentation and no inpainting, so the objects simply were not there when they
scanned.

Consequence for us: our video has the apple and basket on the table, so there is
no trick to copy. Either they are absent from the scan, or they must be masked
and inpainted before the background is trained.

## F3 — Their alignment is a fiducial marker, not a geometric fit

`real-deployment/utils/get_marker2base_aruco.py`, `hand_in_eye_calib.py`: an
ArUco/ChArUco marker on the table gives marker→sim directly, then ICP refines
against a depth camera's cloud. A direct metric measurement.

Everything attempted here instead — disc fits, plane RANSAC, red-cluster
centroids, photometric NCC, ICP — is inference from geometry, good to a few
centimetres at best. `charuco_board_A4.pdf` exists in `/mnt/ssd2/Re3Sim/`.

## F4 — Fitting the table from the splat was worse than not fitting it

Fitting the tabletop out of `bg_lm.ply` (RANSAC plane, then occupancy disc) gave
centre (−0.058, 0.520). Fitting the same table against the **photograph**
instead gave (−0.055, 0.724). Silhouette IoU against the photo's tabletop:

| table pose | source | IoU |
|---|---|---|
| (+0.019, +0.693) | pipeline original | — |
| (−0.058, +0.520) | splat disc fit | 0.380 |
| (−0.055, +0.724) | photo silhouette fit | 0.576 |

The photo fit lands within 3 cm of where the pipeline originally had it. The
splat-based "correction" was a 20 cm error, and it dragged the robot and both
objects with it.

**Lesson: fit against the source frame, not against the splat's geometry.** The
splat's own surfaces are too diffuse to fit a disc to.

Two traps inside the photo fit itself:

* Seeding the tabletop colour from the mesh table's current silhouette
  segments the **carpet**, because the misplaced silhouette overlaps the floor.
  Warm wood versus grey carpet separates cleanly on saturation instead
  (`h ∈ [5,28], s ≥ 55, v ≥ 60` in HSV).
* The table's visual mesh is one geom containing the disc **and** the legs, so
  segmentation cannot isolate the tabletop and IoU saturates near 0.48. Render
  depth, back-project, and keep what lands on the tabletop plane; IoU then
  reaches 0.576.

## F5 — Objects must be fitted individually, not carried with the table

Measured inside `bg_lm.ply`: the photographed apple is a red Gaussian cluster
(7,241 members) at (−0.144, 0.720); the photographed basket is a dark blob
topping out at 0.824 m, centred (−0.003, 0.703). Carrying the objects along with
the table's correction had put them 12.2 cm and 10.8 cm out.

Placing each mesh on its own photograph took foreground alignment from 21.2% to
**24.0% of pixels within 3 cm** (39.8% within 10 cm) — best of the run.

Also, from the photograph: the apple is 67 px across (0.066 m mesh) and clears
the basket by ~29 px ⇒ a real gap of 2.9 cm. They are close in reality; MuJoCo
reports zero contacts between them.

Rest-pose hygiene: objects seated against the *visual* mesh sit 4–5 mm inside
the convex collision hulls. Lifting each body by its own worst penetration gives
0 contacts and 0 penetration at rest, and after 300 steps they settle 0.4 mm
without moving in xy.

## F6 — The ego-view black is capture coverage, not compositing or tilt

Head-pitch sweep, black fraction of the ELP fisheye frame:

| head pitch | black |
|---|---|
| 0° (level) | 88.9% |
| 15° | 84.0% |
| 25° | 79.2% |
| 35° | 74.1% |
| 45° | 69.3% |

Tilting **up** makes it worse: looking down fills the frame with tabletop and
the robot's own body, looking level points at room the splat has no Gaussians
for. So pitch is not the lever.

Splat coverage per camera:

| camera | splat covers | black |
|---|---|---|
| `cap` (the capture pose) | 71.1% | 27.8% |
| `g1_d435i_depth` | 59.9% | 35.9% |

At the robot camera ~39% of the frame is background and essentially all of it is
black. `bg_lm.ply` came from one trimmed clip through Lyra with 16 depth views,
so it barely extends beyond the phone's own field of view along a single orbit.
A 170° fisheye 0.4 m from the table asks for several times more scene than that
reconstruction holds.

Fixes, in order of value: a textured background mesh (F7); retraining the
background on all the videos (the COLMAP model already spans ~10 of them, 450
frames, 349k points); more capture coverage.

## F7 — Background mesh: the bake works, the registration does not

OpenMVS from the existing COLMAP model (450/450 frames registered, 349,306
points). Reusing the dense cloud, `ReconstructMesh --decimate 0.15` then
`TextureMesh --resolution-level 2` gives 374,253 faces and a 4096² atlas in
**1 m 13 s**, versus 41 minutes at full size.

Adding it to the scene as a splat-painted surface cut the ego-view black
**81.5% → 50.7%**.

Three things that bite:

* **`TextureMesh --export-type obj` segfaults in the exporter** in this build
  (OpenMVS 2.3.0), at 2.5M faces/8192 atlas *and* at 374k/4096 — so it is the
  OBJ writer, not memory. `--export-type ply` writes fine. MuJoCo cannot read
  PLY, so convert with trimesh; the material needs `map_Kd` repointed by hand.
* **2.5M faces is the wrong size** for a MuJoCo visual geom re-rendered every
  frame on top of 10.2M Gaussians. Decimate, and crop to the room — MVS
  reconstructs the street through the glass and floor outliers metres below.
* **The mesh must be declared `--splat-geoms`, not added as ordinary geometry.**
  As a plain visual geom it occludes the robot in MuJoCo's depth buffer and the
  foreground mask collapses (24.7% → 3.6%); the robot disappears from every
  render.

### Registration attempts, all failed

| method | result | why it failed |
|---|---|---|
| single shared camera (COLMAP frame 000513 ↔ `cap`) | tabletop within 10 cm | 000513 is 0.4 s before the real capture frame; 0.4 s of handheld motion leaves yaw wrong by tens of degrees |
| photometric NCC vs the source frame, yaw + xyz | plateaus at NCC 0.36 | too weak to trust; MVS texture is blurry so a high NCC is not expected either, which makes the metric hard to read |
| scaled ICP against the splat | collapses to 0.29× | shrinking a cloud into a dense blob trivially lowers every nearest-neighbour distance |
| free rigid ICP | RMS 20.8 cm but **80° residual tilt** | lays the room on its side to chase point distance |
| yaw-constrained ICP (72 starts) | tilt 0°, RMS 34 cm, median NN 48 cm | the two clouds are different sizes — see below |

**Root cause: the scale is wrong.** It came from "frame 513 sits 4.07 colmap
units above the floor", where the floor was the dominant height-histogram peak
of the mesh. That detection is unreliable in this room (glass, large vertical
surfaces, outdoor geometry). Proof: after the transform the mesh spans
z = −7.2 … +3.8 m in world. No room is 11 m tall.

Trying to re-derive scale from the table — the one thing known precisely in
world — also failed: only 320 vertices in the tabletop band near the expected
position, giving a 6.7 cm disc (that is the apple, not the table). The MVS mesh
barely reconstructed the tabletop, so it cannot anchor the scale either.

**Conclusion: geometric registration has run out of signal.** The decisive fix
is F3 — print `charuco_board_A4.pdf`, lay it on the table, and shoot one frame.
That gives the marker→world transform directly and hence the exact
COLMAP→world similarity, which is why Re3Sim does it that way.

## F8 — Deleting objects from the splat is the wrong hammer

Pruning the photographed apple and basket out of the Gaussians
(`scripts/prune_splat.py`, convex hull per body dilated by a margin) removes the
doubling but leaves a hole, because the tabletop *under* an object was never
observed during capture. Visible as a grey smear exactly on the work surface.

Pruning the table as well (23.9% of all Gaussians) guts the floor around it too,
because splat floor Gaussians spread several centimetres in height and fall
inside the table's dilated hull.

RoboSnap does not remove anything from its background, and is right not to: the
mesh is reconstructed from the real object and placed where the real object was,
so it covers its own photograph. Doubling is a symptom of misalignment (F5), not
a design gap.

The exception is **movable** objects over a rollout: once the robot lifts the
apple, the mesh apple moves and the photographed one stays glued to the table.
For that, the non-destructive route is mask + inpaint + retrain, which supplies
the tabletop pixels that were never observed. Pruning cannot.

## F9 — MuJoCo has no fisheye projection

The ELP-USB3DGS1200P01-L180 is ~170° horizontal. A `fovy=170` MuJoCo camera is
not a wide version of the same picture — it is a rectilinear projection that
stretches without bound toward the edges, an image the real lens can never take.

`scripts/fisheye_render.py` renders each eye as a **five-face cube map** (all
that is needed for rays within 90° of forward) and remaps to an equidistant
fisheye, `r = f·θ`. Each face goes through MuGS, so the Gaussian background is
in the result. Fitting `f` from the quoted horizontal FOV:

```
f = 215.7 px/rad from 170° horizontal  ->  127.5° vertical over 480 px
```

The datasheet's 170°/140° pair is not self-consistent for a true equidistant
lens; `f` is fitted from the horizontal number and the resulting vertical
coverage is reported rather than assumed.

Sites must be disabled (`sitegroup[:] = 0`) — the head camera sits 3 cm from its
own mounting site, and at 170° an undisabled site sphere fills a chunk of frame.

## F10 — What was wrong with the stock G1 head cameras

`assets/robots/g1/g1_29dof_with_hand.xml` as shipped:

| camera | parent | pitch | fovy | fovx |
|---|---|---|---|---|
| `ego_view` | torso_link | 45° down | 25.0 | 32.9 |
| `d435i_depth` | torso_link | 45° down | 55.2 | 69.8 |
| `ego_view_left_mono` | left_wrist_roll_link | 53° down | 80.0 | 96.4 |
| `ego_view_right_mono` | **left**_wrist_roll_link | 20° down | 80.0 | 96.4 |

* The D435i was modelled as two cameras 3 cm apart *vertically*. The real module
  has RGB and depth apertures ~1.5 cm apart horizontally on one board.
* Neither FOV matches the datasheet (RGB 69.4°×42.5°, depth 87°×58°).
  `d435i_depth`'s 69.8° horizontal is the *RGB* figure.
* `ego_view_right_mono` is parented to the **left** wrist, with an unnormalised
  quaternion — which is why its pitch differs from its twin. There is no
  right-wrist camera.
* `d435i_depth` produces no depth; it is an ordinary camera.

Replaced with the ELP module (F9). There is **no neck joint** on this robot —
`head_link` is a geom on `torso_link`, and the only joint above the waist is
`waist_pitch_joint`. So "tilting the head" only ever changes the camera
orientation; the shell cannot move. `scripts/set_head_tilt.py` rewrites the
three head cameras' quaternions.

Head-tilt geometry, for a camera 10 cm forward of the neck pivot:

| pitch | forward reach | drop |
|---|---|---|
| 0° | 10.0 cm | 0.0 cm |
| 15° | 9.7 cm | 2.6 cm |
| 45° | 7.1 cm | 7.1 cm |
| 50° | 6.4 cm | 7.7 cm |

## F11 — Operational notes

* **gsplat needs `ninja` from the `robosnap-sim` env on PATH**, or it dies with
  "Ninja is required to load C++ extensions". Prefix every command that touches
  MuGS with `PATH=/mnt/ssd2/conda/envs/robosnap-sim/bin:$PATH`.
* **Long jobs must be `setsid`-detached.** A plain `nohup ... &` is killed with
  its process group when the calling shell is terminated, which silently
  truncated the first two mesh bakes.
* **Check a process by name, not by grepping command strings.** `ps -eo cmd |
  grep -c TextureMesh` counts the leftover `setsid nohup … TextureMesh …`
  wrappers, which reported a bake as "still running" for many minutes after it
  had segfaulted.
* `pkill -f`/`pgrep -f` match the calling shell's own command line; use a
  bracket pattern or filter by pid.
* Camera markers are `site` elements, disabled per-group via
  `MjvOption.sitegroup`, **not** an `mjtVisFlag` — there is no `mjVIS_SITE`.

---

## Open items

1. **Room-mesh registration** — blocked, see F7. Needs the ChArUco frame.
2. **Wide background retrain** — train 3DGS on the full COLMAP model (450
   frames across ~10 videos) rather than the single trimmed clip. Biggest
   available win for F6 and needs no new footage.
3. **Movable-object removal for rollouts** — mask + inpaint + retrain, F8.
4. **ELP stereo baseline is a placeholder** (60 mm). The datasheet does not
   publish the lens spacing; the board is 80 × 16.5 mm. Metric stereo depth
   scales linearly with it, so measure the real module centre-to-centre.
5. **Real head-camera pitch** — currently 20° by request, but the tape
   measurement implies 45–50° (F10). Read the optical axis against a level.

## F12 — a single view cannot determine scale, and the mesh anchor frame is 0.4 s off

Four registration routes for the OpenMVS room mesh, all measured, all rejected:

| route | result | why it fails |
|---|---|---|
| scaled ICP to the splat | collapsed to **0.29x** | shrinking a cloud into a dense blob trivially lowers every nearest-neighbour distance |
| rigid ICP (free rotation) | residual tilt **80 deg** | lays the room on its side to chase point distance; gravity must be locked |
| yaw-locked ICP | RMS 34 cm at best | the splat's Gaussian centres are diffuse, so there is no sharp minimum to find |
| photometric, anchored on the shared frame | NCC identical for every scale | **scale is unobservable from one view** |

That last row is the useful one. Anchoring the mesh so COLMAP frame 000513's
camera coincides with the capture camera gives

    X_world = C_cap + s * R (p - C_513)

so `X_cam = s * R' (p - C_513)` and the projection `u = f x/z` cancels `s`
exactly. Scaling a scene about the camera centre does not change its image.
Scale must come from a known length, never from a single photograph.

Recovering it from the table plane (any mesh point on the tabletop must sit at
world z = 0.713) gave 0.14 with a p25/p75 spread of **1.33** even after
z-buffering to the visible surface. That spread is the tell: the table pixels
are not hitting table geometry, because frame 000513 is 12 frames (0.4 s) from
the real capture frame and 0.4 s of handheld motion is a large rotation. There
is no exactly-shared frame - the trimmed video's frame 0 is original frame 525,
and COLMAP registered 513 and 558, not 525.

**Consequence for the plan.** Registration is not the way in. The splat route
avoids it: `bg_lm.ply` reached world coordinates by two-plane landmark
alignment (floor + tabletop), and that works on a *splat* because a trained
splat has dense clean geometry on both planes. It failed on the MVS mesh only
because MVS put 320 vertices on that tabletop. So retraining on the 450-frame
COLMAP model and re-running the two-plane alignment uses a method that has
already worked once here, instead of one that has failed four times.

Camera pitch across the 450 registered frames, measured (sign of the up axis
fixed from scene mass, not from an arbitrary origin):

    min +5.8   p25 +16.3   median +20.0   p75 +24.9   max +50.9 deg below horizontal
    224 of 450 look down more than 20 deg

The median capture pitch is exactly the head pitch, so the angular coverage for
the ego view is there. What is *not* there is proximity: the phone was 1-2 m
from the table, the robot's eye is 0.4 m, so the tabletop is asked for at 3-5x
the sampling density anything was captured at.

## F13 — the ego-view black was a capture-coverage problem, and training fixed it

The robot's camera rendered a table floating in a void. Every explanation tried
first was wrong:

| tried | result |
|---|---|
| head tilt (0 to 60 deg sweep) | tilting *down* reduced black, but only by filling the frame with the robot's own body: 88.9% at 0 deg to 63.0% at 60 deg |
| Re3Sim's compositing rule (splat painted only on table/ground pixels) | correct in principle, collapses here — the splat has no coverage to paint with |
| a textured room mesh behind the splat | mesh unusable: 2.0 m2 of floor, non-planar walls, no tabletop (F7, F12) |

The cause was simply that `bg_lm.ply` came from **one** trimmed clip through
Lyra — feed-forward, 16 depth views, max-res 640 — so it reconstructs little
more than the phone's own field of view along a single orbit. A 170 deg fisheye
0.4 m from the table asks for several times more scene than that contains.

Retraining on the COLMAP model that already existed - 450 frames spanning ~10
of the videos, 30k iterations of the reference 3DGS densify/prune schedule:

    gs_wide.ply   3,592,082 gaussians   13.2 min   6.7 GB peak
                  PSNR 27.32 dB   SSIM 0.9209   sharpness 0.887
                  38 held-out frames, one fixed resolution
    bg_lm.ply     13-23 dB (Lyra, one clip)

and the coverage measured from each camera:

| background | ego fisheye black | coverage at the robot's eye | at the capture camera |
|---|---|---|---|
| `bg_lm.ply` | 81.5% | 14.9% | 71.1% (27.8% black) |
| `gs_wide_world.ply` | **16.5%** | **81.7%** | **100.0% (0.0% black)** |

The residual 16.5% is mostly outside the 170 deg lens circle rather than
missing data. JPEG payload per eye went 27 kB to 90 kB on the live ZMQ stream,
which is the same fact from another direction: there is content to encode now.

**Why the retrain was the right lever.** Camera pitch across the 450 registered
frames, with the up-axis sign fixed from scene mass rather than an arbitrary
origin:

    min +5.8   p25 +16.3   median +20.0   p75 +24.9   max +50.9 deg below horizontal
    224 of 450 frames look down more than 20 deg

The median capture pitch is *exactly* the head pitch, so the angular coverage
for the ego view was in the footage all along - it was thrown away by using one
clip and a feed-forward reconstructor. What is still not in the footage is
proximity: the phone was 1-2 m from the table and the robot's eye is 0.4 m, so
the tabletop is asked for at 3-5x the sampling density anything was captured at.

**Two fixes that would have bitten any future run.** `train_gs.py` and
`eval_splats.py` both unpacked 8 camera parameters, assuming an OPENCV model;
`colmap image_undistorter` emits a 4-parameter PINHOLE camera, because the
images it writes are already undistorted. Both now accept PINHOLE,
SIMPLE_PINHOLE and OPENCV. `train_gs.py` also now reports SSIM beside PSNR at
every checkpoint - it was already computing SSIM for the loss and never
printing it. PSNR alone can be gained by blurring toward the mean, so a model
can score better while looking worse; here PSNR, SSIM and sharpness all moved
together, which is what makes the gain real detail rather than smoothing.

**Still open.** The transform into world coordinates is good in image space and
wrong in depth: the tabletop lands at z=+0.7133 against 0.7130 expected, but the
rest of the room sits 37 cm in front of the mesh, so only 6% of the robot
survives the depth test. Cause: yaw came from the apple's image position, which
constrains the *direction* to the apple and not the distance along that ray -
and scale sets the distance. Next objective is the splat-vs-mesh depth
agreement `mugs_render` already reports, which cannot be fooled the way image
appearance was.

## F14 — five automated objectives, five different yaws; a person settled it in seconds

With gravity from the camera orbit and scale/position pinned by the table, yaw
was the last free parameter. Every automated objective tried gave a different
answer:

| objective | yaw | why it failed |
|---|---|---|
| NCC, whole frame | 340.25 | table is pinned by construction, so it matches at many yaws and swamps the score |
| NCC, wall band only | 92.50 | mirrored - apple ended up right of the basket |
| red apple's image position | 273.25 | correct side, but only constrains *direction* to the apple |
| splat-vs-mesh depth agreement | 309.25 | best on its own metric, still the wrong wall |
| window brightness centroid | 353.75 | locked onto paper on a noticeboard, mirrored |

The one reliable automatic discriminator was the **mirror test**: in the
photograph the apple sits left of the basket, which holds only from the correct
side of the room. That rules out 92.50 and 353.75 and brackets the answer to
273-310 - but inside that band the metrics disagreed and were fitting noise.

`scripts/tune_splat_yaw.py` exposes yaw as a single knob, blended against the
photograph, re-rendering instantly because the splat stays put and the camera
is inverse-transformed. **Sudhir picked 268.00 deg by eye in seconds**, and it
is visibly correct where all five fits were not: the glass window returns to
where the photograph has it, the backpack to the left, the wall panelling
aligned.

Lesson: single-landmark photometric objectives are unreliable in a room full of
repeated structure. Use them to *bracket* (the mirror test was worth more than
all five fits) and let a person choose inside the bracket.

**Doubled objects.** Chasing each object's convex hull missed the basket,
because the splat's copy of it sits at a different place than the mesh's. The
right shape is a cylinder over the whole tabletop volume - r=0.90 m,
z=0.62..1.20 - since the mesh table and mesh objects own that volume entirely.
Cost 111,025 gaussians, 4.56%. A first attempt at r=0.60 left the splat table's
rim showing as an arc, because the splat renders the table wider than the mesh
does. This only works at all because the wide splat has real geometry behind
those objects; the same prune on `bg_lm.ply` at 14.9% coverage left black holes.

### Final numbers, capture camera and robot eye

| | robot visible | ego fisheye black | doubled objects |
|---|---|---|---|
| `bg_lm.ply` (original) | 25.9% | 81.5% | yes |
| `gs_bg_clean.ply` | **30.2% of a 30.2% mask** | **15.8%** | no |

The robot is now fully visible - every masked pixel survives the depth test -
where before the background was eating 4.3% of it.

## F15 — the tripod was below the floor, and `--keep-below` was protecting it

Eight prune attempts failed to remove the background table's floor frame. Each
cut `z 0.008 .. 0.14` (just above the splat floor at z = -0.0040) and produced a
hole with no improvement.

Rendering `--mode 3dgs_only` against `--mode mujoco_only` from `cap` — two
minutes, and it should have been the first thing done — separated the layers:
the tripod is in the splat, the table's legs and centre post are in the mesh.
Back-projecting the tripod's image footprint through `cap` then located it:

```
tripod gaussians   32,453 in the footprint at depth 1.6-2.4 m from cap
  x  -0.536 .. +0.625
  y  +0.498 .. +1.337
  z  -0.139 .. +0.004    median -0.033      <- BELOW the floor plane
  r   0.00  .. 0.823     from the table centre
```

So every prune had been cutting clean carpet in the band above the floor while
the tripod sat in a thin shell beneath it — and `prune_splat.py --keep-below
0.03` was explicitly preserving that shell. The guard meant to protect the
carpet was protecting the tripod.

The cut that works, colour-gated so same-height carpet survives:

```python
kill = (lum < 0.25) & (X[:,2] > -0.14) & (X[:,2] < 0.012) & (r < 0.95)
# 81,269 of 2,222,113 (3.66%)  ->  gs_bg_notripod.ply, 2,140,844 kept
```

Ego fisheye after the cut: 93.6% covered, 0.0% unfilled inside the lens circle,
so it opens no hole in the robot's view.

**Residue.** A faint grey outline remains and is *not* separable by luminance:
in the tripod's own footprint the remaining gaussians run median lum 0.46-0.58
against reference carpet at median 0.39. The residue is brighter than the carpet
around it, so any threshold that takes it takes the carpet first.

**Why the carpet graft fails** — and not for the reason previously given. The
earlier objection was view-dependent gaussian orientation, but
`apply_splat_transform.py` already drops this PLY to DC-only, so there is no
view-dependent colour to smear. Tried properly (cut the shell colour-blind,
remap a donor ring r 1.50-2.60 inward over the hole) it smears anyway, because
**the remap moves each gaussian's mean but not its covariance**: compressing a
1.1 m ring into a 0.9 m disc stretches every kernel into a streak, worsened by
angular convergence at small radius. A correct graft needs each covariance
rescaled by the local Jacobian, and would still be 25x too sparse — 29,820
donor gaussians against a 753,507-gaussian hole.

Lesson: render each layer alone before theorising about which layer owns a
defect, and check whether a protective guard is protecting the target.
