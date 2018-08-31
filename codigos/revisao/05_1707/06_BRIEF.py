#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by lativ on 28/07/18 at 21:27
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

star = cv.xfeatures2d.StarDetector_create()

brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

kp = star.detect(img)

kp, des = brief.compute(img, kp)

print(brief.descriptorSize())
print(des.shape)

