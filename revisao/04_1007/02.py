#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by lativ on 15/07/18 at 07:23

Copied from 01.py
I will try to use haarcascade_smile.xml to detect the expression in the face region.
"""
import cv2 as cv

# Folder's path
folder_imgs = "/home/lativ/Documents/UFAL/repos/visaocomp/img/"
folder_cascades_xml = '/usr/share/opencv/haarcascades/'

# Cascade objects for face, eyes and smile
face_cascade = cv.CascadeClassifier(folder_cascades_xml + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(folder_cascades_xml + 'haarcascade_eye.xml')
smile_cascade = cv.CascadeClassifier(folder_cascades_xml + 'haarcascade_smile.xml')

# Get first (0) webcam available
cap = cv.VideoCapture(0)

# Now we read frame by frame, detecting the desired regions at each
while cv.waitKey(1) != ord('q'):
    _, img = cap.read()  # Get the frame
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convert to gray-scale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect the points where there are faces

    # For each face region detected
    for (x, y, w, h) in faces:
        # Draw a rectangle around it
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # At this face region, get three views
        # One superior view to detect eyes
        # One inferior view to detect smile
        # And the final to use later to draw the eyes and smile rectangles on top of it.
        roi_gray_sup = gray[y:y+(h//2), x:x+w]
        roi_gray_inf = gray[y+(h//2):y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # Detect the points where there are eyes and smiles
        eyes = eye_cascade.detectMultiScale(roi_gray_sup, 1.3, 5)
        smile = smile_cascade.detectMultiScale(roi_gray_inf, 1.3, 5)

        # For each eye region
        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around it
            cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # For each smile region
        for (sx, sy, sw, sh) in smile:
            # Draw a rectangle around it
            cv.rectangle(roi_color, (sx, h//2 + sy), (sx + sw, h//2 + sy + sh), (0, 0, 255), 2)

    # Show frame with detected regions
    cv.imshow('img', img)

# When 'q' is pressed, the while loop is terminated, the video capture is then released and the window destroyed.
cap.release()
cv.destroyAllWindows()
