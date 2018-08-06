#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by lativ on 28/07/18 at 21:38
"""
import cv2 as cv

def show(img):
    cv.imshow('img', img)
    while True:
        if 0xFF & cv.waitKey(1) == ord('q'):
            break
    cv.destroyAllWindows()

folder_imgs = '/home/lativ/Documents/UFAL/repos/visaocomp/img/'
img = cv.imread(folder_imgs + 'solids.jpg', 0)

orb = cv.ORB_create()

kp = orb.detect(img, None)

kp, des = orb.compute(img, kp)

img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0), flags=0)

show(img2)
