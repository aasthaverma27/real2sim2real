#!/usr/bin/env bash
# Launch opencv_interactive-calibration once the loopback actually has frames.
#
# Two things go wrong when this is run by hand, and both look like a broken
# camera rather than a race:
#   - started before ffmpeg attaches, the device has no format yet and
#     GStreamer dies with "Internal data stream error";
#   - OpenCV's CommandLineParser silently ignores space-separated values, so
#     "-w 8" leaves the board size at its default and the tool complains that
#     the square size must be positive.
# This waits for a readable frame and passes every flag as key=value.
#
#   run_opencv_calib.sh hand_left          # topic name -> output file
#   run_opencv_calib.sh hand_left 8 6 27.94
set -euo pipefail

TOPIC="${1:?usage: run_opencv_calib.sh <topic> [w] [h] [square_mm]}"
W="${2:-8}"; H="${3:-6}"; SZ="${4:-27.94}"
DEV="${DEV:-9}"
OUT="calib/$(echo "$TOPIC" | tr '/' '_')_opencv.xml"
mkdir -p calib

echo "waiting for /dev/video${DEV} to deliver a frame..."
for i in $(seq 1 60); do
  if /usr/bin/python3 -c "
import cv2, sys
c = cv2.VideoCapture(${DEV}, cv2.CAP_V4L2)
ok, f = c.read(); c.release()
sys.exit(0 if ok else 1)" 2>/dev/null; then
    echo "device ready after ${i}s"
    break
  fi
  if [ "$i" = 60 ]; then
    echo "no frames after 60s - is the bridge running on ${TOPIC}?" >&2
    exit 1
  fi
  sleep 1
done

echo "saving to ${OUT}"
echo "REMEMBER: click the Calibration window (or its toolbar save icon) before pressing s"
# Skip GStreamer entirely; its v4l2src does not get on with v4l2loopback.
exec env OPENCV_VIDEOIO_PRIORITY_GSTREAMER=0 \
  opencv_interactive-calibration \
    --ci="${DEV}" -t=chessboard -w="${W}" -h="${H}" --sz="${SZ}" --of="${OUT}"
