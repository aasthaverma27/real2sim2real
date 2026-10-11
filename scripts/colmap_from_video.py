#!/usr/bin/env python3
"""Video -> COLMAP sparse reconstruction, the way Re3Sim does it.

Re3Sim's `re3sim/scripts/reconstruct.py` runs pycolmap (extract_features ->
sequential matching -> incremental_mapping) and then hands the result to the
INRIA 3DGS trainer. This does the COLMAP half against a video instead of a
directory of stills, because that is the input we actually have.

Two things this is careful about that a naive port is not:

**Rotation metadata.** apple.mp4 is portrait with a rotate flag. ffmpeg applies
it, OpenCV does not, so frames pulled with cv2 come out landscape and COLMAP
then fits intrinsics to the wrong aspect. Frames here go through ffmpeg.

**One camera, not one per image.** pycolmap's default camera_mode falls back to
a separate camera per image when EXIF focal length is missing - which it always
is for extracted video frames. That leaves bundle adjustment badly
under-constrained. CameraMode.SINGLE ties every frame to one intrinsic set,
which is physically what happened.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pycolmap


def extract_frames(video: Path, out_dir: Path, count: int, long_side: int) -> int:
    """Evenly sample `count` frames across the whole video via ffmpeg."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(video)],
        capture_output=True, text=True, check=True).stdout.strip())

    # sample by fps rather than seeking per frame - one decode pass, and it
    # keeps the temporal ordering that sequential matching depends on
    fps = count / dur
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video),
         "-vf", f"fps={fps:.6f},scale='if(gt(iw,ih),-2,{long_side})':"
                f"'if(gt(iw,ih),{long_side},-2)'",
         "-q:v", "2", str(out_dir / "%05d.jpg")],
        check=True)

    n = len(list(out_dir.glob("*.jpg")))
    print(f"[frames] {n} frames from {dur:.1f}s of video -> {out_dir}", flush=True)
    return n


def run_colmap(image_dir: Path, colmap_dir: Path, matcher: str) -> None:
    colmap_dir.mkdir(parents=True, exist_ok=True)
    db = colmap_dir / "database.db"
    if db.exists():
        db.unlink()

    print("[colmap] extracting SIFT features ...", flush=True)
    # pycolmap 4.x moved camera_model into reader_options; Re3Sim was written
    # against the 0.x/3.x signature that took it as a top-level kwarg
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV"  # phone lens has real distortion; fit k1..k2
    pycolmap.extract_features(
        database_path=db,
        image_path=image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader,
    )

    print(f"[colmap] {matcher} matching ...", flush=True)
    if matcher == "sequential":
        # loop_detection would close the orbit but needs a vocab tree download;
        # overlap is widened instead so the ends of the sweep still connect
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = 15
        pairing.quadratic_overlap = True
        pycolmap.match_sequential(database_path=db, pairing_options=pairing)
    else:
        pycolmap.match_exhaustive(database_path=db)

    print("[colmap] incremental mapping ...", flush=True)
    sparse = colmap_dir / "sparse"
    sparse.mkdir(exist_ok=True)
    maps = pycolmap.incremental_mapping(
        database_path=db, image_path=image_dir, output_path=sparse)

    if not maps:
        raise SystemExit("[colmap] FAILED: no reconstruction produced")

    # keep the largest model - a sweep that loses tracking yields several
    best = max(maps.items(), key=lambda kv: kv[1].num_reg_images())
    idx, rec = best
    txt = sparse / "text"
    txt.mkdir(exist_ok=True)
    rec.write_text(str(txt))
    print(f"[colmap] {len(maps)} model(s); largest is #{idx} with "
          f"{rec.num_reg_images()} registered images and "
          f"{rec.num_points3D()} sparse points", flush=True)
    for i, r in maps.items():
        print(f"           model {i}: {r.num_reg_images()} imgs, "
              f"{r.num_points3D()} pts", flush=True)

    # the 3DGS trainer expects sparse/0/, so make the winner canonical
    zero = sparse / "0"
    if zero.exists():
        shutil.rmtree(zero)
    zero.mkdir()
    rec.write(str(zero))
    (colmap_dir / "summary.json").write_text(json.dumps({
        "models": {str(i): {"images": r.num_reg_images(),
                            "points": r.num_points3D()} for i, r in maps.items()},
        "chosen": str(idx),
        "registered": rec.num_reg_images(),
        "points3D": rec.num_points3D(),
    }, indent=2))
    # Camera centres, for the manual alignment fallback (tune_splat_yaw.py and
    # apply_splat_transform.py --scale/--yaw read the walk's plane from them).
    (colmap_dir.parent / "colmap_poses.json").write_text(json.dumps(
        {im.name: {"C": [float(x) for x in im.projection_center()]}
         for im in rec.images.values()}, indent=1))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True,
                   help="Work dir; images/ and colmap/ are created inside.")
    p.add_argument("--frames", type=int, default=300,
                   help="How many to sample. MuGS's own guide asks for 50-300.")
    p.add_argument("--long-side", type=int, default=960)
    p.add_argument("--extra-image", type=Path, default=None,
                   help="A still to reconstruct ALONGSIDE the video, named so it "
                        "sorts first. Used for an object photo shot separately "
                        "from an object-free room video: it has no frame 0 to be, "
                        "so it is given a pose the same way every frame gets one. "
                        "Sequential matching pairs it with the frames around the "
                        "start, which is where it was taken from.")
    p.add_argument("--matcher", choices=("sequential", "exhaustive"),
                   default="sequential")
    a = p.parse_args()

    images = a.out / "images"
    n = extract_frames(a.video, images, a.frames, a.long_side)
    if a.extra_image is not None:
        import subprocess as sp
        dst = images / f"00000{a.extra_image.suffix.lower()}"
        sp.run(["ffmpeg", "-v", "error", "-y", "-i", str(a.extra_image),
                "-vf", f"scale='if(gt(iw,ih),-2,{a.long_side})':"
                       f"'if(gt(iw,ih),{a.long_side},-2)'",
                "-q:v", "2", str(dst)], check=True)
        print(f"[frames] + {a.extra_image.name} as {dst.name} "
              f"(object photo, reconstructed with the rest)", flush=True)
        n += 1
    if n < 20:
        raise SystemExit(f"only {n} frames extracted - too few for SfM")
    run_colmap(images, a.out / "colmap", a.matcher)
    print(f"\n[done] sparse model at {a.out / 'colmap' / 'sparse' / '0'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
