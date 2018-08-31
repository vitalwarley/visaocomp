#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 20:36:53 2018

@author: lativ
"""

import cv2
cv2.__version__

vid = cv2.VideoCapture(0)
if vid.isOpened():
    print('webcam ok!')
else:
    print('not found')

cv2.namedWindow('frame', cv2.WINDOW_KEEPRATIO)
while True:
    ret, frame = vid.read()
    cv2.imshow('frame', frame)
    if 0xFF & cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
vid.release()
