#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by lativ on 15/07/18 at 07:23

Copied from 01.py
I will try to use haarcascade_smile.xml to detect the expression in the face region.
"""

import numpy as np
import cv2 as cv

folder_imgs = "/home/lativ/Documents/UFAL/repos/visaocomp/img/"
folder_cascades_xml = '/usr/local/opencv/src/opencv-3.4.1/data/haarcascades/'

face_cascade = cv.CascadeClassifier(folder_cascades_xml + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(folder_cascades_xml + 'haarcascade_eye.xml')
smile_cascade = cv.CascadeClassifier(folder_cascades_xml + 'haarcascade_smile.xml')

img = cv.imread(folder_imgs + 'jl2.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray_sup = gray[y:y + (h//2), x:x+w]
    roi_gray_inf = gray[y + (h//2):y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray_sup, 1.1, 3)
    smile = smile_cascade.detectMultiScale(roi_gray_inf, 1.1, 3)

    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    for (sx, sy, sw, sh) in smile:
        cv.rectangle(roi_color, (sx, h//2 + sy), (sx + sw, h//2 + sy + sh), (0, 0, 255), 2)

cv.imshow('img', img)

while True:
    if 0xFF & cv.waitKey(1) == ord('q'):
        break
cv.destroyAllWindows()

