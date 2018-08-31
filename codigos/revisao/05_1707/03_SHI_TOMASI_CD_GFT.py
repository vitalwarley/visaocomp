#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by lativ on 21/07/18 at 16:17
"""
import numpy as np
import cv2 as cv

folder_imgs = '/home/lativ/Documents/UFAL/repos/visaocomp/img/'
img = cv.imread(folder_imgs + 'solids.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

corners = cv.goodFeaturesToTrack(gray, 25, 0.01, 10)
corners = np.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv.circle(img, (x, y), 3, 255, -1)

cv.imshow('dst', img)
while True:
    if 0xFF & cv.waitKey(1) == ord('q'):
        break
cv.destroyAllWindows()
