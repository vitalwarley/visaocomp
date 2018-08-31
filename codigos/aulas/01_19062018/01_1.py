#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 20:14:41 2018

@author: lativ
"""

import cv2
cv2.__version__

img = cv2.imread('/home/lativ/Documents/UFAL/repos/visaocomp/img/Lenna.png',
                 cv2.IMREAD_COLOR)
cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
cv2.imshow('img', img)

while True:
    if 0xFF & cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
