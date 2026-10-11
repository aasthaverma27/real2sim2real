# Camera calibration: making the sim cameras match the robot's

A VLA trained on simulated images only transfers if the simulated cameras see
the world the way the real ones do: the same field of view, the same mount,
the same framing after every crop and resize. This document covers how the
robot's three cameras were measured and how those numbers reach the MJCF.

| real camera | published topic | published size | MJCF camera(s) |
|---|---|---|---|
| ELP stereo head (dual fisheye module) | `stereo/left`, `stereo/right` | 768 × 432 | `head_pinhole`, `elp_left`, `elp_right` |
| left wrist (IMX335) | `hand_left` | 1280 × 720 | `ego_view_left_mono` |
| right wrist (IMX335) | `hand_right` | 1280 × 720 | `ego_view_right_mono` |

The VLA client resizes every stream to 640 × 480, so the 16:9 sources are
squashed anisotropically to 4:3. The sim renders at the **published**
resolution and the camera server applies the same resize, so the squash is
reproduced rather than approximated by a 4:3 pinhole.

## Intrinsics: checkerboard

We calibrate the **published** stream rather than the raw sensor, so the frame
being measured is the frame the VLA consumes: after the camera server's crop,
before the client's resize. OpenCV's `opencv_interactive-calibration` does the
fit. It only opens video devices, so the ZMQ stream is bridged to a v4l2
loopback device first.

### System requirements

```bash
sudo apt install v4l2loopback-dkms v4l-utils ffmpeg libopencv-dev
# libopencv-dev provides /usr/bin/opencv_interactive-calibration
```

### Board

Print `assets/calib/checkerboard_a4_9x6_25mm.pdf` (9 × 6 inner corners, 25 mm
squares; `scripts/realsim/make_checkerboard.py` makes other sizes) at 100 %
scale. The sheet carries a 100 mm ruler: if it does not measure exactly
100 mm, the printer scaled it and every focal length inherits the error.
**Tape it flat to rigid card** and measure one square with calipers. Loose
paper bows a few millimetres and the solver books the bow as lens distortion:
a hand-held sheet gave `rms 2.27 px`, the same sweep on card gave `0.35 px`.

### Procedure (per camera)

The robot's camera server must be running. `RS_ROBOT_CAMERA` in
`configs/paths.env` points at it (default `tcp://192.168.123.164:5555`, the
G1's onboard computer).

```bash
# 1. a loopback device for the bridge to write into
sudo modprobe v4l2loopback video_nr=9 card_label=robot_cam exclusive_caps=1

# 2. bridge the topic to /dev/video9 (leave running)
scripts/realsim/zmq_to_v4l2.py --topic stereo/right --device /dev/video9

# 3. optional: a live 11-pose checklist that ticks off each pose when held
scripts/realsim/calib_pose_checklist.py --topic stereo/right --board 9x6

# 4. the calibration itself: <topic> <inner-corners-w> <inner-corners-h> <square-mm>
scripts/realsim/run_opencv_calib.sh stereo/right 9 6 25.0
#    -> calib/stereo_right_opencv.xml

# 5. grade the fit and write the yaml + the MuJoCo fovy
scripts/realsim/read_opencv_calib.py calib/stereo_right_opencv.xml \
    --camera head --topic stereo/right --write calib/stereo_right.yaml
```

Repeat for `stereo/left` (`--camera head`), `hand_left` (`--camera wrist_left`)
and `hand_right` (`--camera wrist_right`). Run each camera at least twice.

`read_opencv_calib.py` prints the vertical `fovy` the MJCF camera needs and a
`sim_camera` block. Write that `fovy` into the cameras in
`assets/robots/g1/g1_cad.xml`, then rebuild the robot stages of your scene (see
[assets/robots/g1/README.md](../assets/robots/g1/README.md)).

Aids: `scripts/realsim/record_stream.py` records the stream, so a lost save can
be recovered. `scripts/realsim/opencv_tutorial_calibration.py` is OpenCV's
tutorial fit on recorded frames, as a second opinion.

### What gets rejected silently

- **Tilt 25–35°, never more than 40°.** The tool's `checkLastFrame` drops any
  view tilted more than `badAngleThresh = 40°` about X or Y, without saying so.
- **Start far and flat until `F =` and `RMS =` appear.** Until a first
  calibration exists, that check runs `solvePnP` against a fake `f = 20000`
  matrix, so close-up views are rejected.
- **Start the bridge before the tool**, and **stop the old bridge before
  switching topics**. The loopback has no capture format until a writer
  attaches, and it cannot change frame size (the head is 768 × 432, the wrists
  1280 × 720) while the previous writer holds it.

### Which numbers to trust

The tool's `avg_reprojection_error` and the tutorial's `total error` divide by
the point **count** rather than its square root, so they read about 7× better
than reality. The honest residual is `ret` / `rms_px`.

`cameraMatrix_std_dev` assumes independent views. A hand-held board gives
correlated poses, and here it under-reported by about 20×: two saves a minute
apart differed by 9.6 % in `fx` while each claimed ±0.4 %. **Trust the spread
across repeated runs**, which is why each yaml records `runs_f_px`.

## Our results

`calib/{stereo_left,stereo_right,hand_left,hand_right}.yaml` hold `K`, `D`, the
per-run focal lengths and the `sim_camera` block for our robot (board: 8 × 6
inner corners, 27.94 mm squares, on card).

| topic | fovy per run (deg) | mean | spread |
|---|---|---|---|
| `stereo/left` | 52.961, 53.358 | 53.159 | 0.397 |
| `stereo/right` | 53.537, 53.537 | 53.537 | 0.000 |
| `hand_left` | 60.627, 63.724, 65.768 | 63.373 | **5.140** |
| `hand_right` | 63.811, 63.828 | 63.819 | 0.017 |

Applied to the MJCF: head **53.35°** (mean of the four stereo runs) and both
wrists **63.34°**. Two cross-checks the calibrations had no access to agree:
the two wrist modules, calibrated in separate sessions, match to 0.7°, and the
head eyes' focal ratio (1.017) matches the ratio of their measured fisheye
image circles (1.014).

**Known weakness.** `hand_left`'s runs spread 5.1°, while `hand_right`
repeats to 0.02°. Both wrists currently use 63.34°, which leaves about 0.5°
of error on the right wrist. Re-measure `hand_left` before relying on the wrist
intrinsics further.

The head stream is published pinhole in practice: the camera server's 0.6
centre crop removes the annulus where the fisheye distortion lives, and
straight checkerboard edges in real frames bow by a median of 0.59 px. A plain
MuJoCo pinhole therefore reproduces it. The real wrist streams are raw fisheye
(about 121° horizontal). The camera server renders them as pinholes at the
calibrated fovy by default; `vla_camera_server.py --wrist-projection fisheye`
renders the real equidistant lens instead (`scripts/realsim/wrist_fisheye.py`).

## Extrinsics: CAD plus one measurement

A checkerboard gives the board's pose relative to the camera, never the
camera's pose relative to a robot link, so camera **positions** come from the
CAD of the camera mounts (`assets/robots/g1/g1_cad.xml`, mount parts in
`assets/robots/g1/meshes/cad/`):

| camera | position | orientation |
|---|---|---|
| `ego_view_left_mono` | CAD IMX335 lens barrel | 15° down, 0° yaw |
| `ego_view_right_mono` | CAD IMX335 lens barrel | 25° down, 15° yaw |
| `head_pinhole`, `elp_*` | CAD seat face of the head camera mount | **56.7° down (measured)** |

**The head pitch is measured, not CAD.** The mount is height- and
angle-adjustable, so the CAD shows a nominal 40°. solvePnP against a flat
checkerboard over 159 real frames gives 56.7° (std 2.8°), and at 40° the camera
could not see its own hands, which real frames clearly show.

The CAD export contains the camera parts but not the robot, so two manual
offsets register them to the robot links: the head assembly −57.7 mm and the
wrist assemblies −49 mm, both along +x. As a check against data the CAD never
saw, the fraction of each wrist image covered by the robot's own body is 21 % /
14 % in real frames, 25 % / 27 % with these poses, and was 60 % / 60 % with the
earlier hand-measured ones.

Not done: hand-eye calibration (`cv2.calibrateHandEye` with synchronised joint
states). That would replace the CAD wrist poses and the two manual offsets with
one measured transform per camera.
