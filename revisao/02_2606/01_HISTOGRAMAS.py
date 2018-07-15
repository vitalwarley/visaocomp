#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by lativ on 03/07/18 at 10:06.

From: https://docs.opencv.org/3.4.1/d1/db7/tutorial_py_histogram_begins.html
"""

import cv2 as cv
print(cv.__version__)

# Histogram calculation

# Assuming we are in repos/
img_path = 'visaocomp/img/'
img = cv.imread(img_path + 'Lenna.png', 0)  # Lenna img with IMREAD_GRAYSCALE flag set
hist = cv.calcHist([img], [0], None, [256], [0, 256])
# 'hist' is a 256x1 array, each value corresponds to number of pixels in that image with its corresponding pixel value.

# Histogram calculation in numpy

import numpy as np
hist_np, bins = np.histogram(img.ravel(), 256, [0, 256])

# Faster
hist_bc = np.bincount(img.ravel(), minlength=256)  # OpenCV .calcHist is faster than np.histogram too

# Plotting histograms
#   1. Short way: using matplotlib plotting functions
#   2. Long way: using opencv drawing functions

# Short way
from matplotlib import pyplot as plt

img = cv.imread(img_path + 'Lenna.png', 0)
plt.hist(img.ravel(), 256, [0, 256]); plt.show()

# or with normal plot of matplotlib

img = cv.imread(img_path + 'Lenna.png')  # cv.IMREAD_COLOR, with value 1, is the default flag.
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])

plt.show()

# Long way (code not showed in linked article)
#   code below is adapted from previous classes

# FIXME: not working

cv.namedWindow('img', cv.WINDOW_KEEPRATIO)
histr = [None, None, None]

# with color tuple above
for i, col in enumerate(color):
    histr[i] = cv.calcHist([img], [i], None, [256], [0, 256])

cv.imshow('img', histr[0])

while True:
    if 0xFF & cv.waitKey(1) == ord('q'):
        break
cv.destroyAllWindows()

# --------------------
# Application of Mask

img = cv.imread(img_path + 'Lenna.png', 0)

# create a mask
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:300, 100:400] = 255
masked_img = cv.bitwise_and(img, img, mask=mask)

# Calculate histogram with and without mask
# NOTE: Check third argument for mask
hist_full = cv.calcHist([img], [0], None, [256], [0, 256])
hist_mask = cv.calcHist([img], [0], mask, [256], [0, 256])

# arg for subplot: nrows ncols index, where index go from 1 to nrows * ncols, incremeting in row-major order.
plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0, 256])

plt.show()
