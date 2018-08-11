#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by lativ on 09/08/18 at 19:48

Only began at 10h10, on 11/08/18.
"""
# %% Importing libs
import cv2
import numpy as np

# %% Basic methods

def doNothing():
    pass


def wait_a_little():
    while True:
        if 0xFF & cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()

# %% Importing previous data from 02.py

folder = '/data/Documents/UFAL/repos/visaocomp/aulas/06_0708/'
with np.load(folder + '02_data.npz') as X:
    mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]


# %% Defining basic vars

size = 200
# markerImage = np.zeros((size, size), np.uint8)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
markerId = 10
bborderWidth = 1

# %% Main marker creation

markerImage = cv2.aruco.drawMarker(dictionary, markerId, size, bborderWidth);

# %% Show marker

cv2.imshow('img', markerImage)
wait_a_little()

# %% Marker detection

markerIds, markerCorners, rejectedCandidates = None, None, None

# parameters = cv2.aruco.DetectorParameters_create()

inputImage = cv2.imread('marker.png')
_, _, markerCorners = cv2.aruco.detectMarkers(inputImage, dictionary, markerCorners, markerIds)
# cv2.aruco.detectMarkers(inputImage, dictionary, markerCorners, markerIds, parameters, rejectedCandidates)
# cv2.aruco.estimatePoseSingleMarkers(markerCorners, bborderWidth, mtx, dist, rvecs, tvecs)

# %% Draw detected markers

# cv2.aruco.drawDetectedMarkers(inputImage, markerCorners, markerIds)
cv2.aruco.drawDetectedMarkers(inputImage, markerCorners, markerIds)

# %% Show detected markers

cv2.imshow('img', inputImage)
wait_a_little()