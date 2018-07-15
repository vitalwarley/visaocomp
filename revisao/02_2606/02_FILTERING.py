#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by lativ on 10/07/18 at 20:15

FROM: https://docs.opencv.org/3.4/d4/d13/tutorial_py_filtering.html
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Showing how to use .filter2D
# 2D Convolution (but filter2D actually uses correlation, as specified within)

img = cv.imread('./visaocomp/img/Lenna.png')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

kernel = np.ones((5, 5), np.float32)/25
dst = cv.filter2D(img, -1, kernel)

# -1 in filter2D above is the ddepth, which I didn't understand yet.

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()

# Image blurring (smoothing)

# 1. Averaging

img_jl = cv.imread('./visaocomp/img/jl1.jpg')
img_jl = cv.cvtColor(img_jl, cv.COLOR_BGR2RGB)

blur = cv.blur(img_jl, (5, 5))

plt.subplot(121), plt.imshow(img_jl), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur), plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

# 2. Gaussian Blurring

img_jl2 = cv.imread('./visaocomp/img/jl2.jpg')
img_jl2 = cv.cvtColor(img_jl2, cv.COLOR_BGR2RGB)

gblur = cv.GaussianBlur(img_jl2, (5, 5), 0)  # 0 here means that the sd. will be calc. using the kernel size.

# subplot, remember: line col pos
plt.subplot(121), plt.imshow(img_jl2), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(gblur), plt.title('Blurred with gaussian')
plt.xticks([]), plt.yticks([])
plt.show()

# 3. Median blurring

# Adding noise to an image (from pdi/02.../002_Basic_Operations.py)
img_kw = cv.imread("./visaocomp/img/katw.jpg", cv.IMREAD_COLOR)
img_kw = cv.cvtColor(img_kw, cv.COLOR_BGR2RGB)

noise = np.zeros(img_kw.shape, img_kw.dtype)
cv.randn(noise, 0, 150)
img_kw_with_sp_noise = img_kw + noise

img_kw_filtered_mb = cv.medianBlur(img_kw_with_sp_noise, 5)

plt.subplot(121), plt.imshow(img_kw_with_sp_noise), plt.title('Original with salt & pepper')
plt.subplot(122), plt.imshow(img_kw_filtered_mb), plt.title('Original filtered')
plt.show()

# 4. Bilateral filtering

img_tt1 = cv.imread('./visaocomp/img/texture_tile1.jpg', cv.IMREAD_COLOR)
img_tt1 = cv.cvtColor(img_tt1, cv.COLOR_BGR2RGB)

img_tt1_blurred = cv.bilateralFilter(img_tt1, 9, 75, 75)  # What do the args mean?

plt.subplot(121), plt.imshow(img_tt1), plt.title('Original')
plt.subplot(122), plt.imshow(img_tt1_blurred), plt.title('Blurred with bilateral filtering')
plt.show()

