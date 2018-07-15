#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by lativ on 10/07/18 at 22:19

FROM: https://docs.opencv.org/3.4.1/d0/d86/tutorial_py_image_arithmetics.html
"""

# There is a difference between OpenCV addition and Numpy addition.
# OpenCV addition is a saturated operation while Numpy addition is a modulo operation.

# Image blending: g(x) = (1 - a) f0(x) + af1(x), where a in [0, 1]

import cv2 as cv

img1 = cv.imread('./visaocomp/img/Lenna.png')
img2 = cv.imread('./visaocomp/img/star_bg.png')

dst = cv.addWeighted(img1, 0.7, img2, 0.3, 0)

cv.imshow('dst', dst)
cv.waitKey(0)
cv.destroyAllWindows()

# Bitwise operations

# As imagens são grandes e o threshold não ficou OK.

hacking = cv.imread('./visaocomp/img/hacking2.jpg')
glider = cv.imread('./visaocomp/img/col_glider.jpg')

rows, cols, channels = glider.shape
roi = hacking[0:rows, 0:cols]

glider2gray = cv.cvtColor(glider, cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(glider2gray, 10, 255, cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)

hacking_bg = cv.bitwise_and(roi, roi, mask=mask_inv)

glider_fg = cv.bitwise_and(glider, glider, mask=mask)

dst = cv.add(hacking_bg, glider_fg)
hacking[0:rows, 0:cols] = dst

cv.imshow('res', hacking)
cv.waitKey(0)
cv.destroyAllWindows()
