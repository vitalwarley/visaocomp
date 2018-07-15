#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by lativ on 10/07/18 at 11:03
"""

import cv2

# String contendo o caminho para as imagens usadas no projeto
folder_imgs = "/home/lativ/Documents/UFAL/repos/visaocomp/img/"
# Lendo 'lg.png' como imagem colorida
rgb = cv2.imread(folder_imgs + 'Lenna.png', cv2.IMREAD_COLOR)
# Convertendo a imagem lida acima para escala de cinza
gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

# Partes principais da detecção de faces
cascade_xml = '/usr/local/opencv/src/opencv-3.4.1/data/haarcascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_xml)
faces = face_cascade.detectMultiScale(gray, 1.2, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow('rgb', rgb)
while True:
    if 0xFF & cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
