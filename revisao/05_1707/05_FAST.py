#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by lativ on 22/07/18 at 19:35
"""
import cv2 as cv

def show(img1, img2):
    cv.imshow('with sup', img1)
    cv.imshow('without sup', img2)
    while True:
        if 0xFF & cv.waitKey(1) == ord('q'):
            break
    cv.destroyAllWindows()

folder_imgs = '/home/lativ/Documents/UFAL/repos/visaocomp/img/'
img = cv.imread(folder_imgs + 'solids.jpg', 0)

fast = cv.FastFeatureDetector_create()

kp = fast.detect(img, None)
img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))

print("Threshold: {}".format(fast.getThreshold()))
print("nonMaxSuppressions: {}".format(fast.getNonmaxSuppression()))
print("neighborhood: {}".format(fast.getType()))
print("Total Keypoints with nonMaxSupression: {}".format(len(kp)))

fast.setNonmaxSuppression(0)
kp = fast.detect(img, None)

print("Total keypoints without nonMaxSupression: {}".format(len(kp)))

img3 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))

show(img2, img3)