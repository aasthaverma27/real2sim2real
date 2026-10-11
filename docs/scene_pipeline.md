# Building a scene: from a room video to a photoreal digital twin

One command turns a phone video of a room and one photo of the objects into a
scene that the whole-body controller drives, the photoreal windows render and the
LeRobot exporter records — with **one** background splat over the whole room.
Driving and recording it is covered in [teleop_dataset.md](teleop_dataset.md).

```
room video ─┐
            ├─► COLMAP poses ─► 3DGS training ─► gs_bg_clean.ply ─┐
object image┘                                                     │
     │                                                            ▼
     └─► GUI masks ─► meshes + physics ─► scene_scaled.xml ─► scene_g1.xml  (render)
                                                              └─► scene_wbc.xml (sim)
                                                                       │
                                     WBC ◄── DDS 99 ──────────────────┘
                                      │
                          teleop ─────┤
                                      └──► data exporter ─► LeRobot dataset
```

---

## 0. Capture

**Take the objects off the table, film the room, then put them back and take one
photo.** This is the single biggest decision in the whole pipeline, and the
reference captures get it right by having nothing in them to remove.

It used to be the other way round here, for a reason that no longer holds. The
objects had to be in the video because `gravity_align_scene.py` ties them to the
background through frame 0, and nothing else knew where they went. The `reseat`
stage now places them from a photo by ray-casting onto the splat, so that
constraint is gone — and with it the only argument for filming them.

**An object in the capture cannot be cleanly removed afterwards.** Four ways were
tried on our second room and every one leaves a mark:

| approach | why it fails |
|---|---|
| drop its pixels from the training loss | right for an object that *moves through* a scene, wrong for one standing in it: a basket hides the same patch of tabletop in all 450 frames, so that patch is supervised by nothing, and the gaussians there are never corrected or pruned. Renders as a smear |
| inpaint each frame | a slightly different invented fill in every frame; 3DGS reconciles 450 disagreeing fills by averaging them, which is a blur |
| copy real pixels from other views | the tabletop is planar so the homography is exact, but the poses register it only to within a few pixels — and a few pixels of misregistration is a blur with extra steps |
| build one texture in the surface's plane | a **polished** surface has no single view-independent appearance. Each view sees a different specular sheen, and the median of those is a pale wash |

That last row is the general lesson: the glossier the surface, the less any of
this can work. Two minutes of clearing the table beats all of it.

So the capture is:

1. Move the **robot** out of the room. If it cannot move, mask it — it is not a
   manipulable object, so nothing needs its position.
2. **Take the manipulable objects off the table.** Leave the table itself, and
   everything else that will not move.
3. Film the room in one continuous take.
4. Put the objects back exactly where you want them and take **one photo**,
   roughly from where the video started.
5. Build with `--object-free-video --object-image <that photo>`. The `reseat`
   stage puts the objects onto the splat where the photo says they are.

```bash
bash scripts/build_sim_scene.sh \
    --scene-dir outputs/myroom \
    --room-video room_no_objects.mp4 \
    --object-image objects_on_table.jpg --object-free-video \
    --objects "the round wooden table,the red apple,the woven brown basket" \
    --table-body obj0_round_wooden_table --table-height 0.75 \
    --robot-at <X> <Y> --robot-yaw <RAD>
```

The photo being a separate shot would normally break the alignment, which stands
on frame 0 being in both reconstructions. It does not, because the photo is
handed to COLMAP as one more image, named to sort first: sequential matching
pairs it with the frames around the start of the walk, which is where it was
taken from, and it comes back with a pose in the splat's own frame like
everything else. The splat is then trained on the other frames only, so the one
image that does contain the objects never teaches them to it.

| what it is | where it ends up |
|---|---|
| walls, floor, furniture, the table | the splat — they never move, so this is the best possible way to have them |
| apple, basket, anything grasped | meshes, positioned from the photo, never in the splat at all |
| the robot | not in the capture |

A static **table** does not even need to be a visible mesh. It needs collision so
things rest on it, and the splat already shows it sharply. Name its geoms in
`realsim_view.json` and no renderer will draw the mesh:

```json
{ "exclude_geoms": "floor,ground,room,obj0_round_wooden_table" }
```

Photoreal table, working physics, nothing drawn twice.

If you genuinely cannot re-shoot, `--mask-objects "<prompts>" --surface-prompt
"<the surface>"` is the least-bad option: it paints them out of the frames,
splitting the fill at the surface's edge so the floor is not dragged across the
tabletop. Expect a soft patch where they stood.

What makes a background you can orbit through without seams:

| do | why |
|---|---|
| **one continuous take**, 40–60 s | two captures stitched later leave a blurred seam exactly where they meet. One take, one splat, no seam |
| walk a slow loop, keep the whole room in view | COLMAP needs overlap between consecutive frames; sprinting breaks the chain |
| **change height as you go** — crouch, stand, hold it overhead | the one that gets forgotten, and the one that shows. Our worked example (below) was filmed as a clean 360° orbit covering all 36 ten-degree sectors around the table, at a height that varied by **10 cm over 87 seconds**. The table came out excellent and the floor came out as streaks: a surface seen only at a grazing angle, from a ring of viewpoints all at one height, has no vertical parallax anywhere to fix its depth |
| walk a tight loop **and** a wide one | same reason from the other direction — it gives the floor between them two very different angles |
| keep exposure and white balance locked | auto-exposure makes the same wall two different colours, and training averages them into haze |
| avoid people walking through | anything that moves becomes a smear, because it cannot be explained by one geometry |
| 720p or better, 30 fps | 450 frames at 1200 px long side is the working default |

### The still the objects come from

Frame 0 is doing two jobs at once, and they pull in opposite directions: it is
the **object image** SAM-3D builds meshes from, so it wants to be close and
straight-on, and it is the **first frame of a room sweep**, which usually starts
wherever you happened to be standing.

You do not have to choose. Record the take however it is natural to record it,
note the moment where the objects look best, and pass it:

```bash
--object-frame-time 42
```

The `capture` stage rotates the loop — everything from t=42 s onward, then
everything before it appended to the end. Same frames, same count, same
duration; only the starting point moves. On a capture that orbits back to where
it began, the join lands where the two ends already meet, so the sweep stays
continuous and the loop still closes.

Without the flag the stage still runs, because phone video carries a rotation
flag that ffmpeg honours and other readers do not — and a room reconstructed on
its side is a long detour. It bakes the rotation in, writes
`<scene>/inputs/room.mp4`, and extracts `<scene>/inputs/frame0.png`. There is no
manual ffmpeg step any more; hand the pipeline the file straight off the phone.

**The one thing to plan around:** every object you want as a mesh must be
visible in that frame. SAM-3D builds all of them from that single image, so
anything hidden behind the basket at that instant never becomes an asset.

## 1. Object masks — usually nothing to do

`--objects "the round wooden table,the red apple,the woven brown basket"` is
enough on its own: SAM3 is prompted by text and segments them out of the object
still without anyone touching a GUI. The whole build is then one command.

Open the GUI only when those masks come out wrong — the object pipeline reuses
whatever masks it finds (`SKIP_EXISTING=1`), so a refined mask simply replaces
the automatic one:

```bash
bash scripts/run_gui.sh
```

It is worth checking them once per scene either way. Mask quality is the single
biggest lever on everything downstream — a sloppy mask becomes a sloppy mesh,
which becomes a robot grasping air. They land in `<scene>/sam3d/N.png`, one
RGBA cut-out per object, and a glance at them is usually conclusive.

## 2. Build the scene

```bash
bash scripts/build_sim_scene.sh \
    --scene-dir outputs/myroom \
    --room-video inputs/room.mp4 \
    --object-image inputs/frame0.png \
    --objects "the round wooden table,the red apple,the woven brown basket" \
    --table-body obj0_round_wooden_table --table-height 0.75 \
    --robot-at 0.0 1.37 --robot-yaw -1.62
```

`--robot-at X Y` / `--robot-yaw RAD` are where the robot stands, in the
scene's metric frame (z up, metres, yaw about z). Pick a free spot facing the
table: run the build up to `prune` first (`--dry-run` shows the stages), open
`mujoco/scene_scaled.xml` in `python -m mujoco.viewer`, and read positions off
the floor. `--robot` defaults to `g1_cad`, the G1 with the calibrated CAD
camera rig (`assets/robots/g1/`).

Stages run in order and each is skipped when its output exists:

| stage | produces | notes |
|---|---|---|
| `capture` | `<scene>/inputs/room.mp4`, `frame0.png` | rotation baked in; `--object-frame-time` picks frame 0 |
| `objects` | `fully_refined_foreground.glb` | reuses your GUI masks (`SKIP_EXISTING=1`) |
| `properties` | `physical_properties.json` | mass, friction, restitution by lookup |
| `export` | `mujoco/scene.xml` | meshes + convex collision hulls |
| `rescale` | `mujoco/scene_scaled.xml` | **one real measurement** sets absolute size |
| `colmap` | `roommesh/colmap/sparse/0` | one camera model for all frames; warns if it splits |
| `masks` | `roommesh/masks/*.png` | only in `--mask-objects` mode |
| `inpaint` | `roommesh/images_clean/` | the frames the splat actually trains on |
| `splat` | `gs_wide.ply` | 3DGS, 30k iters, ~13 min, ~7 GB |
| `align` | `gs_bg.ply`, `splat_transform.json` | automatic; stops with advice if it cannot settle the scene |
| `reseat` | objects moved onto the splat | needs `--table-body`/`--table-height` |
| `prune` | `gs_bg_clean.ply` | only in `prune` mode — see below |
| `robot` | `mujoco/scene_g1.xml` | the scene the windows render |
| `wbc` | `mujoco/wbc/scene_wbc.xml` | the scene the sim steps |
| `verify` | pass/fail report | checks the faults that never raise an error |

Useful flags: `--dry-run` prints every command without running it, `--from
STAGE` resumes, `--only STAGE` runs one, `--force` redoes everything.

If the `colmap` stage reports fewer than 85% of frames registered, or more than
one model, the room did not reconstruct as one piece — the splat would then be
trained on half of it. Re-run that stage with `--matcher exhaustive --force`,
which is slower but ties an orbit's two ends together regardless of how far
apart they are in time.

### Keeping objects out of the background

An object that is in the room when you film it gets **baked into the splat**, and
then it is drawn twice: once as background, once as the mesh the physics actually
moves. Push the real apple and its ghost stays behind on the table.

There are three ways out, and they are not equally good:

| mode | flag | what happens | cost |
|---|---|---|---|
| **painted out of the frames** | `--mask-objects "red apple, brown woven basket"` | SAM3 segments them in every frame; the hole is filled with plausible surface before training | invented pixels over a small area — **use this** |
| object-free capture | `--object-free-video` | film the room with the objects already removed | nothing at all, but then nothing ties the objects to the room |
| dropped from the loss | *(train_gs.py `--masks`)* | those pixels contribute no gradient | **only correct for an object that moves** — see below |
| prune afterwards | *(default)* | cut a volume out of a splat that already learned the objects | leaves a hole and a visible rim |

**Do not drop a stationary object's pixels from the loss.** It is the obvious
idea and it fails in a way that looks like a worse version of the problem it
solves. It is sound when the object *moves through* the scene: the patch it hides
in one view is visible in the next, so the optimiser still learns what is behind
it. A basket standing on a table hides the *same* patch of tabletop in all 450
frames. Masked out of the loss, that patch is constrained by nothing at all —
the gaussians there receive no gradient, so they are never corrected and never
pruned, and whatever the initialisation left renders as a smear lying on the
table.

Painting the objects out removes the hole instead of creating one. Every frame
gets plausible tabletop, the views disagree slightly, and 3DGS averages that
into a smooth surface — which on uniform wood is indistinguishable from the real
thing.

**Name the surface too.** An object standing on a table hides part of the table
*and* part of whatever is behind it, so its mask straddles the table's far rim.
Fill that in one go and the floor gets dragged down onto the tabletop in a grey
wedge — worse than the object was. `--surface-prompt "round wooden table"`
segments the table as well, so the hole is split at the rim and each side is
filled only from its own surface.

One subtlety that bites: the segmenter returns the table *without* the objects
on it, so the table mask has object-shaped holes and every pixel to be filled
looks like "not table". The pipeline closes those enclosed holes first — the
patch under a basket is table. A notch in the silhouette's edge, where the
basket stands proud of the far rim, is not enclosed and correctly stays outside.

Pruning is the fallback for footage you cannot re-shoot. It cannot be clean:
the gaussians it removes were fitted to explain pixels that included the object,
so their neighbours are already contaminated, and the cut edge shows it.

This is also what the reference footage does — a scan with no manipulable objects
in it, and the objects added as assets afterwards. Re3Sim's reconstruction script
contains no mask, no segmentation and no inpainting for exactly that reason
(`docs/real2sim_experiments.md` F2). Two minutes of moving objects off the table
before you press record buys a background nothing else can.

### Scale, alignment and reseating

**Scale.** Reconstruction is scale-free, so `--table-height` (or any one
measured length on a named body) is what turns proportions into metres.

Measure a *second* object and pass it too — it is the only way to find out
whether the first measurement was the right one, since nothing in the factor
ever looked at it:

```bash
--check-body obj2_woven_brown_basket --check-size 0.245 0.19
```

Read the two errors separately. Both small means the factor is right. One small
and one large means the factor is fine and that object's *mesh* came out the
wrong shape — a single head-on view is thin evidence for a depth the camera
never saw. Both large means the anchor measurement is wrong.

**Splat alignment.** This used to be the manual step. It no longer is, because
frame 0 is in *both* reconstructions — it is the object still and the first
video frame — so the camera that took it is an exact shared reference, and the
rotation and position of the splat follow from it with nothing to fit.

Everything that camera does not settle is read off **the splat**, never off the
object reconstruction:

| what | from | rather than |
|---|---|---|
| which way is up | the splat's floor plane, RANSAC over ~60,000 points | a plane fitted to one VGGT depth map, which was **6.4° off** in the worked example |
| the lens | COLMAP's bundle adjustment over 450 views | VGGT's guess for the same camera, **11%** different |
| scale | the splat's own floor-to-tabletop, against `--table-height` | VGGT depths, which are **2.6×** out here |

Each of those is checkable, and the stage prints the check. The one that catches
everything is the camera path: push all 450 COLMAP cameras through the finished
transform and they must look like a person walking. Before the levelling fix
they wandered ±0.16 m in height; after it, ±0.02 m.

If it cannot settle a scene it says so and stops (exit code 3), with what to
check: usually the capture (a split COLMAP model, or an object photo taken far
from where the video starts). Automated *fitting* of the yaw was tried and gave
five different answers (`docs/real2sim_experiments.md` F14); this stage does not
fit anything. A known similarity can be passed with `--splat-scale`,
`--splat-yaw` and `--world-table`, resuming with `--from align`.

**Reseating.** Aligning the splat is only half of it: the objects are still
where the object reconstruction put them, which is not where they are. VGGT
recovers depth up to a factor that varies along the viewing ray, so they come
out the right size at the wrong distance — in the worked example, the table landed 0.55 m
from its true position. You see this immediately as **two tables**, one
photoreal and one solid, a couple of handspans apart.

The `reseat` stage fixes it with the information already present: each object's
mask says which direction it lies in from frame 0's camera, and the splat says
where the surface it rests on is.

The table is found by fitting its **rim**, not by averaging its surface. The
average is wrong in a way that looks fine — the masked-out apple and basket
leave a hole in the tabletop, and the mean of what remains is dragged off centre
by a third of the table. The rim is a circle, and the centre that puts the most
points at a common radius is unambiguous: 7,176 points over 100% of the circle,
here.

Then check the two against each other. The mesh table came out 0.795 m across
and the splat's measured 0.759 m — 5%, so the scale is sound. A large
disagreement means the *proportions* are wrong rather than the placement: size
was anchored on the measured height alone, so a round table reconstructed from
one near-head-on view keeps its error in the diameter. Measure the diameter too
and rebuild.


## Verifying a scene

```bash
python scripts/realsim/verify_scene.py --scene-dir outputs/myroom
```

Runs automatically as the last pipeline stage. Every check is there because the
fault it catches presents as "the stack runs but the robot behaves oddly", never
as an error:

| check | what it caught |
|---|---|
| base joint is joint 0, joints 1..43 hinges | the WBC reads `qpos[joint_id + 6]` |
| all 43 wire joints present | the 51-float mirror packet cannot be filled |
| hand actuators in thumb-first order | every finger executing another finger's command |
| finger joints have armature and damping | a bare finger link is ~2e-4 kg·m²; the WBC's torque slams it between its limits at 100 rad/s |
| `frame.json` yaw matches the built scene | the stored value is rounded — rebuilding from it moves the robot 0.2° off its own splat |
| ego camera, background splat, movable objects | the windows and the recorder have something to read |

Exit code is the number of failures, so it gates a build.

## What the pieces are

| file | what it is |
|---|---|
| `gs_wide.ply` | the trained splat, still in COLMAP's frame |
| `gs_bg.ply` | the same splat, placed in the scene's metric frame |
| `gs_bg_clean.ply` | tabletop volume cleared — **this is what the windows render** |
| `mujoco/scene_scaled.xml` | objects only, metric |
| `mujoco/scene_g1.xml` | objects + robot, namespaced `g1_` — the render scene |
| `mujoco/wbc/scene_wbc.xml` | robot-first re-merge — the scene the sim steps |
| `mujoco/wbc/scene_wbc.frame.json` | the frame shift the windows undo |

Why two scenes: the controller requires bodies to keep their own names
(`pelvis`, `torso_link`), the base joint to be joint 0, and the hinges to be
1..N with nothing in front. Attaching a robot *into* a scene breaks all three,
so the WBC scene is merged the other way round, with the robot as parent, and
the windows undo the resulting frame shift.


## Environments

| env | has | used for |
|---|---|---|
| `robosnap-sim` | mujoco, gsplat, MuGS, cv2, zmq | rendering, scene building |
| `robosnap-lyra` | pycolmap, gsplat training | `train_gs.py` |
| `robosnap-asset` / `-sam3` | SAM3, SAM-3D, VGGT | masks and meshes |
| `.venv_sim` (`scripts/setup_external.sh`) | unitree_sdk2py, gear_sonic | the DDS sim |

Every entry point sources `scripts/_bootstrap.sh`, which reads
`configs/paths.env` and puts `robosnap-sim/bin` on `PATH` — gsplat JIT-compiles
CUDA on first use and needs `ninja` from that env.

**VRAM ceiling.** `RS_VRAM_CEILING_MIB` in `configs/paths.env` (default 31000)
is a hard cap: `build_sim_scene.sh` and `start.sh` both refuse to start a stage
that would cross it, which matters on a shared card. Never run two heavy stages
at once.

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

Refusing is better than failing, but it still leaves you watching the card. Park
the stage behind the wait instead, and it starts the moment there is room:

```bash
bash scripts/wait_for_vram.sh 9000 -- bash scripts/build_sim_scene.sh --from masks ...
```

Roughly what the heavy stages want: `objects` ~12 GB, `masks` ~3 GB (SAM3 at
`--proc-long-side 1024`; ~5 GB at native resolution, which is enough to OOM on a
busy card and buys nothing, since the mask is dilated afterwards anyway),
`splat` ~7 GB. `colmap` is mostly CPU and fits in about 2 GB.


## A worked example

A room with a round table, an apple and a woven basket, from one 87 s handheld
orbit with the objects left on the table (so the `--mask-objects` route).

```bash
bash scripts/build_sim_scene.sh \
    --scene-dir outputs/apple_room \
    --room-video room_orbit.mp4 \
    --object-frame-time 42 \
    --objects "the round wooden table,the red apple,the woven brown basket" \
    --mask-objects "red apple, brown woven basket" \
    --surface-prompt "round wooden table" \
    --table-body obj0_round_wooden_table --table-height 0.75 \
    --check-body obj2_woven_brown_basket --check-size 0.245 0.19 \
    --robot-at 0.016 0.425 --robot-yaw 1.5429

scripts/realsim/start.sh --hold --paused --scene outputs/apple_room
```

| stage | result | read it as |
|---|---|---|
| `capture` | 2619 frames, 87.3 s, unchanged | rotating the loop is lossless; only the start moved |
| `colmap` | **450/450 registered, 1 model**, 91,950 points | the room reconstructed as one piece |
| `masks` | objects 1.8% of frame, table 7.7%, empty on 0 frames | every frame knows where both are |
| `inpaint` | 450 frames repaired, 2.1% of each | the splat never sees a hole |
| `splat` | 1,087,976 gaussians, PSNR 33.46 dB, SSIM 0.965 | |
| `align` | floor was 6.45° off level, corrected to 0.32° | camera path went from ±0.16 m to ±0.02 m in height |
| `reseat` | table moved 0.55 m; rim 0.749 m vs mesh 0.795 m | 6% — the two reconstructions agree |
| `verify` | all checks pass | |

Its `realsim_view.json` also excludes the table mesh from drawing:

```json
{ "exclude_geoms": "floor,ground,room,obj0_round_wooden_table" }
```

Its collision geoms are group 3 and were never drawn anyway, so objects still
rest on it — but the splat now shows the *real* table, wood grain and stickers
included, instead of a flat orange disc drawn on top of it.

Three things this capture taught, all visible in the result:

**Vary your height.** The orbit covered all 36 sectors around the table at
1.25–1.35 m. Table and walls are excellent; the floor near the camera ring is
streaked, because no view ever saw it from a different height and nothing
constrains its depth.

**Measure more than one length.** The basket's short side matched to 3% and its
long side to 20%: the *scale* is right and the *mesh* came out square, because
one near-head-on view says almost nothing about depth. The table has the same
problem more mildly, 6% in its diameter.

**Never trust one reconstruction's opinion of another's frame.** Gravity, the
lens and the scale all came from VGGT originally and all three were wrong — by
6.4°, 11% and a factor of 2.6. Each is now read off the splat or off COLMAP,
and each is checked.

