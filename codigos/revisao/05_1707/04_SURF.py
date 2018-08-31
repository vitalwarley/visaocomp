#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by lativ on 22/07/18 at 08:38
"""
import numpy as np
import cv2 as cv

def show(img):
    cv.imshow('dst', img)
    while True:
        if 0xFF & cv.waitKey(1) == ord('q'):
            break
    cv.destroyAllWindows()

folder_imgs = '/home/lativ/Documents/UFAL/repos/visaocomp/img/'
img = cv.imread(folder_imgs + 'castle.jpg', 0)

surf = cv.xfeatures2d.SURF_create(5000)

kp, des = surf.detectAndCompute(img, None)

img2 = cv.drawKeypoints(img, kp, None, (255, 0, 0), 4)

show(img2)

# Now without orientation
surf.setUpright(True)

kp = surf.detect(img, None)
img2 = cv.drawKeypoints(img, kp, None, (255, 0, 0), 4)

show(img2)

# If you want to extend the descriptor size
print(surf.descriptorSize())
surf.setExtended(True)
kp, des = surf.detectAndCompute(img, None)
print(surf.descriptorSize())
print(des.shape)

# Now we would do matching..
