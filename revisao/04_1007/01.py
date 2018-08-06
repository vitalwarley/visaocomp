#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by lativ on 15/07/18 at 07:23

FROM: https://docs.opencv.org/3.4/d7/d8b/tutorial_py_face_detection.html
"""

import numpy as np
import cv2 as cv

# Folder's path
folder_imgs = "/home/lativ/Documents/UFAL/repos/visaocomp/img/"
folder_cascades_xml = '/usr/share/opencv/haarcascades/'

face_cascade = cv.CascadeClassifier(folder_cascades_xml + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(folder_cascades_xml + 'haarcascade_eye.xml')

img = cv.imread(folder_imgs + 'jl1.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

cv.imshow('img', img)
while True:
    if 0xFF & cv.waitKey(1) == ord('q'):
        break
cv.destroyAllWindows()

# Detectando a boca como sendo olhos para 'jl1.jpg'
# Detectando nariz e orifícios do nariz como sendo olhos para 'katw.jpg'
# Detectando orifício esquerdo do nariz como sendo olho para 'ts1.jpg'
# Detectando orifícios do nariz como sendo olhos para 'woman1.jpg'
# O que fazer nesse caso? O que mudo?
