#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 09:27:28 2018

@author: lativ
"""

import cv2

img = cv2.imread('/home/lativ/Documents/UFAL/repos/visao/img/Lenna.png',
                 cv2.IMREAD_COLOR)
cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
hist_img = cv2.calcHist(img)
cv2.imshow(hist_img)

while True:
    if 0xFF & cv2.waitKey(1) == ord('q'):
        break
    
cv2.destroyAllWindows()

# Not running properly.
