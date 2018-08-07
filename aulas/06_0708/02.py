#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by lativ on 07/08/18 at 11:16

Task:

    Efetuar a calibração de uma câmera usando imagens com padrão circular.
"""

# %% Importing libs

import numpy as np
import cv2
import glob

# %% Basic utils
def wait_or_press_q():
    while True:
        if 0xFF & cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

# %% Main variables

folder = 'symmetric_circular_grid/'

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((5*6, 3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:5].T.reshape(-1,2)

objpoints = []
imgpoints = []

images = glob.glob(folder + '*.png')

# %% Setup and drawing

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findCirclesGrid(gray, (6, 5), None)

    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        img = cv2.drawChessboardCorners(img, (6, 5), corners2, ret)

        cv2.imshow('img', img)
        cv2.waitKey(1000)

# Showing only the last one
wait_or_press_q()

# %% Calibration

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# %% Undistortion

img = cv2.imread(folder + '25.png')
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# %% Using cv2.undistort()

dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

#diff = cv2.absdiff(img, dst)

cv2.imshow('img distorted', img)
cv2.imshow('img calibrated', dst)
# cv2.imshow('differences', diff)
wait_or_press_q()

# %% Using remapping

mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

# diff = cv2.absdiff(img, dst)

cv2.imshow('img distorted', img)
cv2.imshow('img calibrated', dst)
#cv2.imshow('differences', diff)
wait_or_press_q()

#%% Save data

np.savez('02_data', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
