#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by lativ on 07/08/18 at 11:51
"""

# %% Importing libs

import cv2
import numpy as np
import glob

# %% Importing previous data from 02.py

folder = '/data/Documents/UFAL/repos/visaocomp/aulas/06_0708/'
with np.load(folder + '02_data.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

# %% Basic utils
def wait_or_press_q():
    while True:
        if 0xFF & cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img

# FIXME: cv2.line w error
def draw_cube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j].ravel()), (255, 0, 0), 3)

    img = cv2.drawContours(img, [imgpts[4::]], -1, (0, 0, 255), 3)

    return img

# %% Basic variables

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((5*6, 3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:5].T.reshape(-1,2)

axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

# %% Load, search, refine (if found), draw
folder = ('/data/Documents/UFAL/repos/' +
          'material_repos/camera_calibration_API/examples/' +
          'example_images/symmetric_grid/')

images = glob.glob(folder + '*.png')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findCirclesGrid(gray, (6, 5), None)

    if ret == True:

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw(img, corners2, imgpts)
        cv2.imshow('img', img)
        cv2.waitKey(1000)

# Showing only the last one
wait_or_press_q()


# %% Render a cube

# FIXME: couldn't draw cube because of erros in tutorial and lack of time to fix

folder = ('/data/Documents/UFAL/repos/' +
          'material_repos/camera_calibration_API/examples/' +
          'example_images/symmetric_grid/')

images = glob.glob(folder + '*.png')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findCirclesGrid(gray, (6, 5), None)

    if ret == True:

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw_cube(img, corners2, imgpts)
        cv2.imshow('img', img)
        cv2.waitKey(1000)

# Showing only the last one
wait_or_press_q()


