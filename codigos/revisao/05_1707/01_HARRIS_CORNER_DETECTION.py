#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by lativ on 21/07/18 at 15:50

From: https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
"""

import numpy as np
import cv2 as cv

folder_imgs = '/home/lativ/Documents/UFAL/repos/visaocomp/img/'
img = cv.imread(folder_imgs + 'chessboard.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv.cornerHarris(gray, 2, 3, 0.04)

dst = cv.dilate(dst, None)

img[dst > 0.01 * dst.max()] = [0, 0, 255]

cv.imshow('dst', img)
while True:
    if 0xFF & cv.waitKey(1) == ord('q'):
        break
cv.destroyAllWindows()
