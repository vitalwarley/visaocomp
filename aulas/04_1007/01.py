#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by lativ on 10/07/18 at 11:03
"""

import cv2

path_img = "/home/lativ/Documents/UFAL/repos/visaocomp/img/lg.png"
rgb = cv2.imread(path_img, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
cascade_xml = '/usr/local/opencv/src/opencv-3.4.1/data/haarcascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_xml)
faces = face_cascade.detectMultiScale(gray, 1.2, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(rgb, (x, y), (w, h), (255, 0, 0), 2)

cv2.imshow('rgb', rgb)
while True:
    if 0xFF & cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
