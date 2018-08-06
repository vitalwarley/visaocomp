# header?

import cv2 as cv
import numpy as np

folder_imgs = "/home/lativ/Documents/UFAL/repos/visaocomp/img/"
img = cv.imread(folder_imgs + 'castle.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift = cv.xfeatures2d.SIFT_create()
kp = sift.detect(gray, None)

img = cv.drawKeypoints(gray, kp, img)

cv.imshow('img', img)
cv.waitKey(0)