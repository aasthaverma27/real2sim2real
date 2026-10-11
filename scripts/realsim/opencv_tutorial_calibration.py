#!/usr/bin/env python3
"""OpenCV's Camera Calibration tutorial, verbatim.

Source: doc/py_tutorials/py_calib3d/py_calibration/py_calibration.markdown
        (opencv/opencv, tag 4.13.0) - rendered at
        https://docs.opencv.org/4.13.0/dc/dbb/tutorial_py_calibration.html

The code below is the tutorial's, character for character. The only values
changed are the two the tutorial itself requires you to set for your own
board: the pattern size (the tutorial's demo board is (7,6); ours is (8,6)
inner corners) and the image glob.

Note the tutorial's objp is in units of ONE SQUARE, not metres. That is
correct and needs no fixing: scaling the target scales the board's distance,
not the camera matrix, so mtx and dist come out the same either way.

    opencv_tutorial_calibration.py <image_dir>
"""
import sys

import numpy as np
import cv2 as cv
import glob

PATTERN = (8, 6)                      # tutorial: (7,6)
IMAGE_GLOB = sys.argv[1] if len(sys.argv) > 1 else '*.jpg'
SHOW = '--show' in sys.argv           # tutorial always shows; needs a display

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((PATTERN[1]*PATTERN[0],3), np.float32)
objp[:,:2] = np.mgrid[0:PATTERN[0],0:PATTERN[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(IMAGE_GLOB)

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, PATTERN, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        if SHOW:
            cv.drawChessboardCorners(img, PATTERN, corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)

if SHOW:
    cv.destroyAllWindows()

print(f"{len(objpoints)} of {len(images)} images had a {PATTERN[0]}x{PATTERN[1]} board")

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("ret (rms):", ret)
print("mtx:\n", mtx)
print("dist:\n", dist)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )

# --- what the numbers mean for the sim, derived from mtx alone -------------
h, w = gray.shape
fovx = 2*np.degrees(np.arctan(0.5*w/mtx[0,0]))
fovy = 2*np.degrees(np.arctan(0.5*h/mtx[1,1]))
print(f"\nresolution  {w}x{h}")
print(f"fx {mtx[0,0]:.2f}  fy {mtx[1,1]:.2f}  cx {mtx[0,2]:.2f}  cy {mtx[1,2]:.2f}")
print(f"fovx {fovx:.3f} deg   fovy {fovy:.3f} deg   <- MuJoCo fovy")
