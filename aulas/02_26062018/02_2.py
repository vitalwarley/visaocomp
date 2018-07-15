#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 10:46:19 2018

@author: lativ
"""

import numpy as np
import cv2 as cv

img = cv.imread('/home/lativ/Documents/UFAL/repos/visaocomp/img/Lenna.png',
                cv.IMREAD_COLOR)
kernel = np.ones((5, 5), np.float32)/25
dst = cv.medianBlur(img, 5)
cv.namedWindow('img', cv.WINDOW_KEEPRATIO)
cv.imshow('img', dst)

while True:
    if 0xFF & cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()
