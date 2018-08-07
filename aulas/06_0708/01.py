#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by lativ on 07/08/18 at 09:30

Manipulação de imagens:
 1. Translação
 2. Rotação
 3. Redimensionamento, escalonamento e interpolação.
 4. Imagens de pirâmide
 5. Homografia

"""
# %% Importing libs
import cv2
import numpy as np

# %% Basic variables

img = cv2.imread('Lenna.png')

# %% Basic utils
def wait_or_press_q():
    while True:
        if 0xFF & cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


# %% Translation

rows, cols = img.shape[:2]

"""
M =
[ 1 0 100
  0 1  50]
"""
M = np.float32([[1, 0, 100], [0, 1, 50]])
dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow('original', img)
cv2.imshow('translated', dst)
wait_or_press_q()

# %% Rotation

rows, cols = img.shape[:2]

M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1)
dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow('original', img)
cv2.imshow('rotated', dst)
wait_or_press_q()


# %% Scaling
# TODO: missing two things

height, width = img.shape[:2]
res = cv2.resize(img, (width//2, height//2), interpolation=cv2.INTER_AREA)
# res = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

cv2.imshow('original', img)
cv2.imshow('scaled', res)
wait_or_press_q()

# %% Image pyramids down

lower_reso_1 = cv2.pyrDown(img)
lower_reso_2 = cv2.pyrDown(lower_reso_1)
lower_reso_3 = cv2.pyrDown(lower_reso_2)

cv2.imshow('original', img)
cv2.imshow('lower 1', lower_reso_1)
cv2.imshow('lower 2', lower_reso_2)
cv2.imshow('lower 3', lower_reso_3)
wait_or_press_q()

# %% Image pyramids up from lowest

higher_reso_1 = cv2.pyrUp(lower_reso_3)
higher_reso_2 = cv2.pyrUp(higher_reso_1)
higher_reso_3 = cv2.pyrUp(higher_reso_2)

cv2.imshow('original lower 1', lower_reso_3)
cv2.imshow('higher 1', higher_reso_1)
cv2.imshow('higher 2', higher_reso_2)
cv2.imshow('higher 3', higher_reso_3)
wait_or_press_q()

# %% Affine transformation

rows, cols, ch = img.shape

pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

M = cv2.getAffineTransform(pts1, pts2)

dst = cv2.warpAffine(img, M, (cols, rows))


# %% Homography

import matplotlib.pyplot as plt

MIN_MATCH_COUNT = 10

img1 = cv2.imread('bottle.jpg', 0)           # queryImage
img2 = cv2.imread('bottle_in_scene.jpg', 0)  # trainImage

# Get it right: rotate and reescale
rows, cols = img2.shape[:2]
#M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1)
#img2 = cv2.warpAffine(img2, M, (cols, rows))

img2 = cv2.resize(img2, (cols//4, rows//4), interpolation=cv2.INTER_AREA)
rows, cols = img1.shape[:2]
img1 = cv2.resize(img1, (cols//4, rows//4), interpolation=cv2.INTER_AREA)
# res = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

#cv2.imshow('bottle', img1)
#cv2.imshow('in scene', img2)
#wait_or_press_q()

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

good = []

for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    draw_params = dict(matchColor = (0, 255, 0), singlePointColor = None,
                       matchesMask = matchesMask,
                       flags = 2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    plt.imshow(img3, 'gray'), plt.show()

else:

    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None
