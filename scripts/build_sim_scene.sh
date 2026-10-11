#!/usr/bin/env bash
# Capture -> teleop-ready simulation, in one command.
#
# This is step 2 of two. Step 1 is the GUI, where you make the object masks:
#
#     bash scripts/run_gui.sh
#
# Then this script takes the masks plus a room video and produces a scene the
# whole-body controller, the photoreal windows and the data exporter can all
# drive, with ONE background splat covering the whole room - not two captures
# stitched together, which is what leaves a blurred seam where they meet.
#
#     bash scripts/build_sim_scene.sh \
#         --scene-dir outputs/myroom \
#         --room-video inputs/room.mp4 \
#         --object-image inputs/frame0.png \
#         --objects "the round wooden table,the red apple,the woven brown basket" \
#         --table-body obj0_round_wooden_table --table-height 0.75
#
# Then:  scripts/realsim/start.sh --vla --scene outputs/myroom
#        ...and the WBC / teleop / exporter commands it prints.
#
# Stages, each skipped when its output already exists (--force redoes all,
# --from STAGE resumes, --only STAGE runs one):
#
#   objects     masks + image -> meshes, collision hulls, mass/friction
#   properties  physical properties lookup
#   export      MuJoCo scene.xml from the object meshes
#   rescale     one measured length -> metric scene_scaled.xml
#   colmap      room video -> COLMAP poses (single capture, whole room)
#   splat       COLMAP -> 3DGS background, optimised against the frames
#   align       place the splat in the scene's metric frame
#   prune       cut the objects out of the splat  (only in the default mode)
#   robot       attach the robot -> scene_g1.xml   (the render scene)
#   wbc         re-merge robot-first -> wbc/scene_wbc.xml  (the sim scene)
#   verify      check the scene against every fault that has broken silently
#
# Keeping the objects out of the background - they are drawn twice otherwise,
# once as splat and once as the mesh the physics moves:
#
#   --mask-objects "<prompts>"   segment them in every frame and PAINT THEM OUT
#                                before training, so the splat sees a continuous
#                                surface where they stood. Headless. Add
#                                --surface-prompt "round wooden table" so the
#                                fill cannot drag the floor across the table's
#                                edge.
#   --object-free-video          film the room WITHOUT them. Nothing to remove,
#                                nothing to see, and free - but then nothing ties
#                                the objects to the room either.
#   --inpaint-objects <masks>    paint them out of the frames first. Needs
#                                per-frame GUI masks, and invents pixels.
#   (default)                    prune them out afterwards - lossy, leaves a rim.
#
# Why COLMAP + 3DGS and not the feed-forward path: Lyra predicts a depth map per
# view and stacks them, with no loss anywhere, so views that disagree are
# averaged rather than reconciled and an orbiting capture smears. Training
# against the frames is what gives a room you can orbit through cleanly.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
. "$ROOT/scripts/_bootstrap.sh" || exit 1
cd "$ROOT"

SIM="$RS_PY_SIM"
LYRA="$RS_PY_LYRA"
SAM3="$RS_PY_SAM3"

SCENE=""; ROOM_VIDEO=""; OBJECT_IMAGE=""; OBJECTS=""; OBJECT_FRAME_TIME=""
TABLE_BODY=""; TABLE_HEIGHT=""; CHECK_BODY=""; CHECK_SIZE=""
SPLAT_SCALE=""; SPLAT_YAW=""; WORLD_TABLE=""
ROBOT="g1_cad"; ROBOT_AT=""; ROBOT_YAW=""
# How the manipulable objects are kept out of the background splat. An object
# baked into the splat is drawn twice - once as background, once as the mesh the
# physics actually moves - so it has to be absent BEFORE training, or cut out
# afterwards. Cutting out is the lossy option: it leaves a hole and the residue
# around its rim. See docs/scene_pipeline.md "Keeping objects out of the background".
BG_MODE=prune           # prune | object-free | inpaint | mask
MASKS=""; MASK_PROMPTS=""; SURFACE_PROMPT=""
FRAMES=450; LONG_SIDE=1200; ITERS=30000; HOLDOUT=12; VRAM_CAP=26
# Sequential matching assumes the capture is a walk. A capture that orbits back
# to where it started needs the two ends tied together or the room comes out as
# two half-models; quadratic overlap usually bridges it, exhaustive always does.
MATCHER=sequential
# How far reseat may move a body before it assumes it has matched the wrong
# thing rather than found a badly placed one.
MAX_RESEAT=0.9
FORCE=0; FROM=""; ONLY=""; DRY=0
VRAM_CEILING="${RS_VRAM_CEILING_MIB:-31000}"

usage() { sed -n '2,/^set -uo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --scene-dir)     SCENE="$2"; shift 2 ;;
    --room-video)    ROOM_VIDEO="$2"; shift 2 ;;
    --object-image)  OBJECT_IMAGE="$2"; shift 2 ;;
    --object-frame-time) OBJECT_FRAME_TIME="$2"; shift 2 ;;
    --objects)       OBJECTS="$2"; shift 2 ;;
    --table-body)    TABLE_BODY="$2"; shift 2 ;;
    --table-height)  TABLE_HEIGHT="$2"; shift 2 ;;
    --check-body)    CHECK_BODY="$2"; shift 2 ;;        # second object...
    --check-size)    CHECK_SIZE="$2 $3"; shift 3 ;;     # ...and its measured LONG SHORT

    --splat-scale)   SPLAT_SCALE="$2"; shift 2 ;;
    --splat-yaw)     SPLAT_YAW="$2"; shift 2 ;;
    --world-table)   WORLD_TABLE="$2"; shift 2 ;;   # "x y z", from the tuner
    --robot)         ROBOT="$2"; shift 2 ;;
    --object-free-video) BG_MODE=object-free; shift ;;
    --inpaint-objects)   BG_MODE=inpaint; MASKS="$2"; shift 2 ;;
    --mask-objects)      BG_MODE=mask; MASK_PROMPTS="$2"; shift 2 ;;
    --surface-prompt)    SURFACE_PROMPT="$2"; shift 2 ;;
    --robot-at)      ROBOT_AT="$2 $3"; shift 3 ;;
    --robot-yaw)     ROBOT_YAW="$2"; shift 2 ;;
    --frames)        FRAMES="$2"; shift 2 ;;
    --matcher)       MATCHER="$2"; shift 2 ;;
    --max-reseat)    MAX_RESEAT="$2"; shift 2 ;;
    --long-side)     LONG_SIDE="$2"; shift 2 ;;
    --iters)         ITERS="$2"; shift 2 ;;
    --force)         FORCE=1; shift ;;
    --from)          FROM="$2"; shift 2 ;;
    --only)          ONLY="$2"; shift 2 ;;
    --dry-run)       DRY=1; shift ;;
    -h|--help)       usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 2 ;;
  esac
done

[ -n "$SCENE" ] || { echo "need --scene-dir" >&2; usage 2; }

STAGES="capture objects properties export rescale colmap masks inpaint splat align reseat prune robot wbc verify"
STARTED=0
[ -n "$FROM" ] || STARTED=1

want() {   # want <stage>: should this stage run at all?
  local s="$1"
  if [ -n "$ONLY" ]; then [ "$ONLY" = "$s" ] && return 0 || return 1; fi
  if [ "$STARTED" = 0 ]; then
    [ "$FROM" = "$s" ] && STARTED=1 || return 1
  fi
  return 0
}

have() {   # have <file>: output already there and --force not given?
  [ "$FORCE" = 0 ] && [ -e "$1" ]
}

run() {    # run <command...>: honour --dry-run, fail loudly
  if [ "$DRY" = 1 ]; then printf '    %q' "$@"; echo; return 0; fi
  "$@" || { echo "[pipeline] FAILED: $*" >&2; exit 1; }
}

step() { printf '\n=== [pipeline] %s ===\n' "$*"; }
skip() { printf '    (skipped, %s exists - --force to redo)\n' "$1"; }

vram_guard() {   # vram_guard <need MiB> <what>
  [ "$DRY" = 1 ] && return 0
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)
  echo "[pipeline] VRAM ${used} MiB used, $2 needs ~$1 MiB, ceiling ${VRAM_CEILING} MiB"
  if [ $((used + $1)) -gt "$VRAM_CEILING" ]; then
    echo "[pipeline] ABORT: that would cross the ceiling. The card is shared:" >&2
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader >&2
    echo "[pipeline] wait for it to free, then re-run with --from $3" >&2
    exit 1
  fi
}

set -a; . "$ROOT/configs/auto_pipeline.env" 2>/dev/null || true; set +a

mkdir -p "$SCENE"
echo "[pipeline] scene   $SCENE"
if   [ -n "$ONLY" ]; then STAGE_DESC="only $ONLY"
elif [ -n "$FROM" ]; then STAGE_DESC="$FROM -> end"
else                     STAGE_DESC="all (${STAGES// /, })"; fi
echo "[pipeline] stages  $STAGE_DESC"

# ---------------------------------------------------------------- capture ----
# Everything downstream reads two files that both come out of the one take:
# the room video, and the still the objects are built from. They are not
# independent - gravity_align_scene.py ties the objects to the background with
#
#     background_camera = pad_extrinsic(cameras["w2c_render"][0])
#
# i.e. it assumes the object still IS frame 0 of the room video. So the still
# cannot be a separate photo, and it cannot be a frame from the middle either.
#
# Which is a problem when the best view of the objects is not where you happened
# to start recording. --object-frame-time fixes that by rotating the loop: the
# chosen instant becomes frame 0 and the footage before it is appended to the
# end. For a capture that orbits back to where it started, the join lands where
# the two ends already meet, so nothing is lost and the loop still closes.
#
# With no --object-frame-time this still normalises the video, which is worth a
# stage on its own: phone video carries a rotation flag that ffmpeg honours and
# other readers do not, and a room reconstructed on its side is a long detour.
PREP="$SCENE/inputs"
[ -f "$PREP/room.mp4" ]   && ROOM_VIDEO="$PREP/room.mp4"
[ -f "$PREP/frame0.png" ] && [ -z "$OBJECT_IMAGE" ] && OBJECT_IMAGE="$PREP/frame0.png"

if want capture; then
  step "capture: bake the rotation, put the object view at frame 0"
  if have "$PREP/room.mp4"; then skip "inputs/room.mp4"
  else
    [ -n "$ROOM_VIDEO" ] || { echo "need --room-video" >&2; exit 2; }
    # in a dry run the capture stage has not written it yet
    [ "$DRY" = 1 ] || [ -f "$ROOM_VIDEO" ] || {
      echo "no such video: $ROOM_VIDEO" >&2; exit 2; }
    mkdir -p "$PREP"
    if [ -n "$OBJECT_FRAME_TIME" ]; then
      echo "    rotating the take so t=${OBJECT_FRAME_TIME}s leads"
      run ffmpeg -v error -y -i "$ROOM_VIDEO" -filter_complex \
        "[0:v]trim=start=${OBJECT_FRAME_TIME},setpts=PTS-STARTPTS[a];[0:v]trim=end=${OBJECT_FRAME_TIME},setpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=1[v]" \
        -map "[v]" -an -c:v libx264 -crf 16 -pix_fmt yuv420p "$PREP/room.mp4"
    else
      run ffmpeg -v error -y -i "$ROOM_VIDEO" -an -c:v libx264 -crf 16 \
        -pix_fmt yuv420p "$PREP/room.mp4"
    fi
  fi
  ROOM_VIDEO="$PREP/room.mp4"
  if have "$PREP/frame0.png"; then skip "inputs/frame0.png"
  else run ffmpeg -v error -y -i "$ROOM_VIDEO" -vframes 1 "$PREP/frame0.png"
  fi
  [ -n "$OBJECT_IMAGE" ] || OBJECT_IMAGE="$PREP/frame0.png"
fi

# With an object-free video the object photo is a SEPARATE shot, so it is not
# frame 0 of anything and the shared camera the alignment stands on does not
# exist. Handing the photo to colmap as one more image restores it: named to
# sort first, sequential matching pairs it with the frames either side of the
# start, which were taken from the same spot, and it comes back with a pose in
# the splat's own frame. The splat is then trained on the other frames only, so
# the one image that does contain the objects never teaches them to it.
OBJ_FRAME=""
if [ "$BG_MODE" = object-free ] && [ -n "$OBJECT_IMAGE" ]; then
  OBJ_FRAME="$SCENE/roommesh/images/00000.${OBJECT_IMAGE##*.}"
fi

# ---------------------------------------------------------------- objects ----
if want objects; then
  step "objects: masks + image -> meshes, collision hulls, mass and friction"
  if have "$SCENE/fully_refined_foreground.glb"; then
    skip "fully_refined_foreground.glb"
  else
    [ -n "$OBJECT_IMAGE" ] || { echo "need --object-image" >&2; exit 2; }
    # SKIP_EXISTING=1 is what makes this reuse the masks you refined in the GUI
    # instead of re-running SAM3 over them.
    if [ "$DRY" = 1 ]; then
      echo "    INPUT_IMAGE=$OBJECT_IMAGE OUTPUT_DIR=$SCENE OBJECT_PROMPTS='$OBJECTS' \\"
      echo "      bash scripts/run_object_pipeline.sh"
    else
      INPUT_IMAGE="$OBJECT_IMAGE" OUTPUT_DIR="$SCENE" OBJECT_PROMPTS="$OBJECTS" \
        SKIP_EXISTING=1 bash "$ROOT/scripts/run_object_pipeline.sh" \
        || { echo "[pipeline] FAILED: object pipeline" >&2; exit 1; }
    fi
  fi
fi

# ------------------------------------------------------------- properties ----
if want properties; then
  step "properties: mass, friction, restitution"
  if have "$SCENE/physical_properties.json"; then skip "physical_properties.json"
  else run "$SIM" scripts/estimate_properties.py --scene-dir "$SCENE" --mode lookup
  fi
fi

# ----------------------------------------------------------------- export ----
if want export; then
  step "export: MuJoCo scene.xml"
  if have "$SCENE/mujoco/scene.xml"; then skip "mujoco/scene.xml"
  else MUJOCO_GL=egl run "$SIM" scripts/export_mujoco.py --scene-dir "$SCENE" --static table
  fi
fi

# ---------------------------------------------------------------- rescale ----
# Reconstruction is scale-free, so absolute size comes from ONE real measurement.
if want rescale; then
  step "rescale: one measured length -> metres"
  if have "$SCENE/mujoco/scene_scaled.xml"; then skip "mujoco/scene_scaled.xml"
  else
    [ -n "$TABLE_BODY" ] && [ -n "$TABLE_HEIGHT" ] || {
      echo "need --table-body and --table-height (the one real measurement" >&2
      echo "that sets absolute scale; bodies are named in $SCENE/mujoco/scene.xml)" >&2
      exit 2; }
    # One measurement sets the scale; a second, unrelated one is the only way
    # to find out whether it was the right measurement.
    CHECK=()
    [ -n "$CHECK_BODY" ] && [ -n "$CHECK_SIZE" ] && \
      CHECK=(--basket-body "$CHECK_BODY" --check-basket $CHECK_SIZE)
    MUJOCO_GL=egl run "$SIM" scripts/rescale_scene.py \
        --scene "$SCENE/mujoco/scene.xml" \
        --output "$SCENE/mujoco/scene_scaled.xml" \
        --table-body "$TABLE_BODY" --table-height "$TABLE_HEIGHT" "${CHECK[@]}"
  fi
fi

# ------------------------------------------------------------- clean video ----
# Masking and inpainting the objects out of the FRAMES means the splat never
# learns them, so there is nothing to cut out later and no rim to see.
if want colmap && [ "$BG_MODE" = inpaint ]; then
  step "clean video: inpaint the objects out before the splat ever sees them"
  CLEAN_VIDEO="${ROOM_VIDEO%.*}_clean.mp4"
  if have "$CLEAN_VIDEO"; then skip "$(basename "$CLEAN_VIDEO")"
  else
    [ -n "$MASKS" ] || { echo "--inpaint-objects needs the GUI mask dir" >&2; exit 2; }
    run "$SIM" scripts/make_clean_background_video.py \
        --video "$ROOM_VIDEO" --masks "$MASKS" --output "$CLEAN_VIDEO"
  fi
  ROOM_VIDEO="$CLEAN_VIDEO"
fi

# ----------------------------------------------------------------- colmap ----
if want colmap; then
  step "colmap: room video -> poses (ONE capture, whole room)"
  if have "$SCENE/roommesh/colmap/sparse/0"; then skip "roommesh/colmap/sparse/0"
  else
    [ -n "$ROOM_VIDEO" ] || { echo "need --room-video" >&2; exit 2; }
    # in a dry run the capture stage has not written it yet
    [ "$DRY" = 1 ] || [ -f "$ROOM_VIDEO" ] || {
      echo "no such video: $ROOM_VIDEO" >&2; exit 2; }
    # pycolmap lives in robosnap-lyra, not robosnap-sim. Running this under
    # $SIM fails with ModuleNotFoundError after the frames are already extracted.
    run "$LYRA" scripts/colmap_from_video.py \
        --video "$ROOM_VIDEO" --out "$SCENE/roommesh" \
        --frames "$FRAMES" --long-side "$LONG_SIDE" --matcher "$MATCHER" \
        ${OBJ_FRAME:+--extra-image "$OBJECT_IMAGE"}
    # A split reconstruction is the failure that looks like success: the stage
    # exits 0, the splat trains on half the room, and the other half is missing.
    if [ "$DRY" = 0 ] && [ -f "$SCENE/roommesh/colmap/summary.json" ]; then
      "$SIM" - "$SCENE/roommesh/colmap/summary.json" "$FRAMES" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1])); want = int(sys.argv[2])
got = s["registered"]; frac = got / max(want, 1)
print(f"[pipeline] colmap registered {got}/{want} frames ({frac:.0%}), "
      f"{len(s['models'])} model(s)")
if frac < 0.85 or len(s["models"]) > 1:
    print("[pipeline] WARNING: the capture did not reconstruct as one room.\n"
          "           Re-run with --matcher exhaustive --force --only colmap\n"
          "           (slower, but it ties an orbit's two ends together).")
PYEOF
    fi
  fi
fi

# ------------------------------------------------------------------ masks ----
# The objects are in the room while it is being filmed, so they land in the
# splat and get drawn twice: once as background, once as the mesh the physics
# moves. Push the real apple and its ghost stays on the table.
#
# Segmenting them per frame and dropping those pixels from the loss is the
# cleanest answer available. Nothing is painted, nothing is invented and nothing
# is cut: the optimiser simply never sees the object, and reconstructs the table
# under it from the views where it is not in the way - which an orbit has plenty
# of. Only the contact patch is unconstrained.
MASK_DIR="$SCENE/roommesh/masks"
SURF_DIR="$SCENE/roommesh/masks_surface"
CLEAN_DIR="$SCENE/roommesh/images_clean"
if want masks && [ "$BG_MODE" = mask ]; then
  step "masks: segment the objects out of every frame (SAM3, text prompts)"
  if have "$MASK_DIR"; then skip "roommesh/masks"
  else
    [ -n "$MASK_PROMPTS" ] || { echo "--mask-objects needs prompts" >&2; exit 2; }
    vram_guard 6000 "SAM3 per-frame segmentation" masks
    run "$SAM3" \
        scripts/mask_frames_sam3.py \
        --images "$SCENE/roommesh/images" --out "$MASK_DIR" \
        --prompts "$MASK_PROMPTS"
  fi
  # The surface the objects stand on, segmented separately. Without it the fill
  # has no idea where the tabletop ends and will happily smear the floor across
  # it, because an object's mask straddles that edge.
  if [ -n "$SURFACE_PROMPT" ]; then
    if have "$SURF_DIR"; then skip "roommesh/masks_surface"
    else
      run "$SAM3" \
          scripts/mask_frames_sam3.py \
          --images "$SCENE/roommesh/images" --out "$SURF_DIR" \
          --prompts "$SURFACE_PROMPT" --dilate 0
    fi
  fi
fi

# ---------------------------------------------------------------- inpaint ----
# Why paint the objects out rather than drop their pixels from the loss: a
# basket on a table never moves, so it hides the SAME patch of tabletop in all
# 450 frames. Masked out of the loss, that patch is supervised by nothing, the
# gaussians sitting there get no gradient and are neither corrected nor pruned,
# and the splat ends up with a smear on the table - worse than the ghost the
# mask was there to prevent. Loss masking is right for something that moves
# through the scene; this is right for something standing in it.
if want inpaint && [ "$BG_MODE" = mask ]; then
  step "inpaint: paint the objects out of the frames before the splat sees them"
  if have "$CLEAN_DIR"; then skip "roommesh/images_clean"
  else
    SURF=(); [ -d "$SURF_DIR" ] && SURF=(--surface-masks "$SURF_DIR")
    run "$SIM" scripts/inpaint_frames.py \
        --images "$SCENE/roommesh/images" --masks "$MASK_DIR" \
        --out "$CLEAN_DIR" --dilate 8 "${SURF[@]}"
  fi
fi

# ------------------------------------------------------------------ splat ----
if want splat; then
  step "splat: 3DGS trained against the frames (~13 min at 30k iters)"
  if have "$SCENE/gs_wide.ply"; then skip "gs_wide.ply"
  else
    vram_guard 7000 "3DGS training" splat
    IMGS="$SCENE/roommesh/images"
    [ "$BG_MODE" = mask ] && [ -d "$CLEAN_DIR" ] && IMGS="$CLEAN_DIR"
    # train_gs skips any registered image whose file it cannot find, so leaving
    # the object photo out of this directory is all it takes to keep the
    # objects out of the splat while keeping its pose in the reconstruction
    if [ -n "$OBJ_FRAME" ] && [ "$DRY" = 0 ]; then
      IMGS="$SCENE/roommesh/images_nobj"
      mkdir -p "$IMGS"
      for q in "$SCENE"/roommesh/images/*; do
        case "$(basename "$q")" in 00000.*) continue ;; esac
        [ -e "$IMGS/$(basename "$q")" ] || ln -s "$(readlink -f "$q")" "$IMGS/"
      done
    fi
    MASKARG=()
    run "$LYRA" scripts/train_gs.py \
        --colmap "$SCENE/roommesh" --image-dir "$IMGS" \
        --out "$SCENE/gs_wide.ply" --iters "$ITERS" --long-side "$LONG_SIDE" \
        --holdout "$HOLDOUT" --vram-limit-gb "$VRAM_CAP" "${MASKARG[@]}"
  fi
fi

# ------------------------------------------------------------------ align ----
# The splat comes out in COLMAP's arbitrary frame. align_splat_auto.py places it
# without any fitting (see below). Only when that cannot settle a scene does the
# stage stop and hand over to the interactive tuner - a person matched the yaw
# in seconds where five automated objectives disagreed
# (docs/real2sim_experiments.md F14).
if want align; then
  step "align: place the splat in the scene's metric frame"
  if have "$SCENE/gs_bg.ply"; then skip "gs_bg.ply"
  elif [ -z "$SPLAT_SCALE" ] || [ -z "$SPLAT_YAW" ]; then
    # The automatic route. It works because frame 0 is in BOTH reconstructions,
    # so the camera that took it is a shared, exact reference - and because
    # everything it cannot read off that camera, it reads off the splat itself
    # rather than off VGGT: gravity from the floor plane (tens of thousands of
    # points, against one depth map), the lens from colmap's bundle adjustment,
    # and scale from the splat's own floor-to-tabletop against --table-height.
    [ -n "$TABLE_HEIGHT" ] || { echo "align needs --table-height" >&2; exit 2; }
    if [ "$DRY" = 1 ]; then
      printf '    %q' "$LYRA" scripts/align_splat_auto.py --scene-dir "$SCENE" \
        --table-height "$TABLE_HEIGHT"; echo
    fi
    if [ "$DRY" = 1 ] || "$LYRA" scripts/align_splat_auto.py --scene-dir "$SCENE" \
         --table-height "$TABLE_HEIGHT"; then
      run "$SIM" scripts/apply_splat_transform.py \
          --ply "$SCENE/gs_wide.ply" --out "$SCENE/gs_bg.ply" \
          --transform "$SCENE/splat_transform.json" --min-opacity 0.02
    else
      cat >&2 <<EOF

[pipeline] STOP - the automatic alignment could not settle this scene.

Most often the capture is the cause: check the colmap stage registered one
model with most frames (re-run with --matcher exhaustive --force --only colmap)
and that the object photo was taken near where the video starts.

If you know the similarity by other means, pass it and resume:

  --splat-scale <S> --splat-yaw <DEG> --world-table "<x> <y> <z>" --from align

scripts/tune_splat_yaw.py --scene-dir $SCENE lets you set the yaw by eye
against the object photo; it needs a camera named 'cap' at the photo's pose in
mujoco/scene_scaled.xml.
EOF
      exit 3
    fi
  else
    WT=(); [ -n "$WORLD_TABLE" ] && WT=(--world-table $WORLD_TABLE)
    run "$SIM" scripts/apply_splat_transform.py \
        --ply "$SCENE/gs_wide.ply" --out "$SCENE/gs_bg.ply" \
        --scale "$SPLAT_SCALE" --yaw "$SPLAT_YAW" --min-opacity 0.02 "${WT[@]}" \
        --poses "$SCENE/roommesh/colmap_poses.json"
  fi
fi

# ----------------------------------------------------------------- reseat ----
# The splat and the object reconstruction disagree about where things are, and
# the splat wins: colmap triangulated from 450 views that had to agree, while
# VGGT inferred depth from one. Left alone the disagreement shows as two tables,
# one photoreal and one solid, a couple of handspans apart.
if want reseat; then
  step "reseat: move the mesh objects onto the splat"
  if [ -z "$TABLE_BODY" ] || [ -z "$TABLE_HEIGHT" ]; then
    echo "    (skipped, needs --table-body and --table-height)"
  else
    MUJOCO_GL=egl run "$SIM" scripts/reseat_objects_to_splat.py \
        --scene-dir "$SCENE" --table-body "$TABLE_BODY" \
        --table-height "$TABLE_HEIGHT" --max-move "$MAX_RESEAT"
  fi
fi

# ------------------------------------------------------------------ prune ----
if want prune; then
  if [ "$BG_MODE" != prune ]; then
    step "prune: skipped - the background has no objects in it ($BG_MODE)"
    cp -n "$SCENE/gs_bg.ply" "$SCENE/gs_bg_clean.ply" 2>/dev/null || true
  else
  step "prune: cut the objects out of the splat (lossy - see docs/scene_pipeline.md)"
  if have "$SCENE/gs_bg_clean.ply"; then skip "gs_bg_clean.ply"
  else
    run "$SIM" scripts/prune_splat.py \
        --ply "$SCENE/gs_bg.ply" --mjcf "$SCENE/mujoco/scene_scaled.xml" \
        --out "$SCENE/gs_bg_clean.ply" --keep-below 0.03
  fi
  fi
fi

# ------------------------------------------------------------------ robot ----
if want robot; then
  step "robot: attach to the scene -> scene_g1.xml (what the windows render)"
  if have "$SCENE/mujoco/scene_g1.xml"; then skip "mujoco/scene_g1.xml"
  else
    [ -n "$ROBOT_AT" ] && [ -n "$ROBOT_YAW" ] || {
      echo "need --robot-at X Y and --robot-yaw RAD (where the robot stands)." >&2
      echo "Pick a free spot facing the table: open the scene and read the" >&2
      echo "floor position under the cursor (double-click, then Ctrl+right-drag):" >&2
      echo "  $SIM -m mujoco.viewer --mjcf $SCENE/mujoco/scene_scaled.xml" >&2
      exit 2; }
    MUJOCO_GL=egl run "$SIM" scripts/add_robot.py \
        --scene-dir "$SCENE" --scene-xml "$SCENE/mujoco/scene_scaled.xml" \
        --robot "assets/robots/g1/${ROBOT}.xml" \
        --at $ROBOT_AT --yaw "$ROBOT_YAW" --strip-rangefinders \
        --output "$SCENE/mujoco/scene_g1.xml"
  fi
fi

# -------------------------------------------------------------------- wbc ----
# The controller is strict: bodies keep their own names, the base joint must be
# joint 0 and the hinges 1..N with nothing in front. So the merge runs the other
# way round - robot as parent, scene attached into it.
if want wbc; then
  step "wbc: robot-first re-merge -> wbc/scene_wbc.xml (what the sim steps)"
  if have "$SCENE/mujoco/wbc/scene_wbc.xml"; then skip "mujoco/wbc/scene_wbc.xml"
  else
    [ -n "$ROBOT_AT" ] && [ -n "$ROBOT_YAW" ] || {
      echo "need the same --robot-at / --robot-yaw used for the robot stage" >&2
      exit 2; }
    MUJOCO_GL=egl run "$SIM" scripts/realsim/build_wbc_scene.py \
        --scene-dir "$SCENE" --robot "$ROBOT" \
        --scene-xml "$SCENE/mujoco/scene_g1.xml" --strip-prefix g1_ \
        --at $ROBOT_AT --yaw "$ROBOT_YAW"
  fi
fi

# ----------------------------------------------------------------- verify ----
# Everything checked here has broken silently before: a scrambled hand order
# moves the wrong fingers, fingers without armature flail, and a rounded yaw in
# frame.json moves the robot off its own splat. None of them raise an error.
if want verify; then
  step "verify: check the scene against the faults that do not announce themselves"
  MUJOCO_GL=egl run "$SIM" scripts/realsim/verify_scene.py --scene-dir "$SCENE"
fi

[ "$DRY" = 1 ] && { echo; echo "[pipeline] dry run - nothing was executed"; exit 0; }

cat <<EOF

=== [pipeline] scene ready: $SCENE ===

  render scene   $SCENE/mujoco/scene_g1.xml
  sim scene      $SCENE/mujoco/wbc/scene_wbc.xml
  background     $SCENE/gs_bg_clean.ply

Drive it - see docs/teleop_dataset.md for the full command list:

  scripts/realsim/start.sh --vla --scene $SCENE
  # then the WBC, teleop and exporter commands it prints
EOF
