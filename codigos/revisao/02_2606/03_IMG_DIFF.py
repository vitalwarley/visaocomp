#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by lativ on 10/07/18 at 21:08

FROM: https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/

NOT EXACTLY AS IT WAS SHOWED IN CLASS.
"""

from skimage.measure import compare_ssim
import argparse
import imutils
import cv2 as cv

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--first', required=True,
                help='first input image')
ap.add_argument('-s', '--second', required=True,
                help='second')
args = vars(ap.parse_args())

imageA = cv.imread(args['first'])
imageB = cv.imread(args['second'])

grayA = cv.cvtColor(imageA, cv.COLOR_BGR2GRAY)
grayB = cv.cvtColor(imageB, cv.COLOR_BGR2GRAY)

(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype('uint8')
print("SSIM: {}".format(score))

thresh = cv.threshold(diff, 0, 255,
                      cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                       cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

for c in cnts:
    (x, y, w, h) = cv.boundingRect(c)
    cv.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv.imshow("Original", imageA)
cv.imshow("Modified", imageB)
cv.imshow("Diff", diff)
cv.imshow("Thresh", thresh)
cv.waitKey(0)
