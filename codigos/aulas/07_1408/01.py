#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by lativ on 14/08/18 at 10:07

ConvNets etc.
"""
# %% Importing libs

import keras
print(keras.__version__)
import matplotlib.pyplot as plt
import numpy as np

# %% Instantiating a small convnet

from keras import layers
from keras import models

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))

model.summary()

# %% Adding a classifier on top of the convnet

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# %% Training the convenet on MINIST images

from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
idx = 10
plt.imshow(255 - train_images[idx], cmap = 'gray')
plt.xlabel('Class = ' + str(train_labels[idx]))
plt.show()

# %% Training ...

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# %% Train and save model
model.fit(train_images, train_labels, epochs=5, batch_size=64)
model.save('model-today-5.1.h5')
#from keras.models import load_model
#model = load_model('model-today-5.1.h5')

# Missing parts

# %% Investigate some test samples

pred_labels = model.predict_classes(test_images)
pred_scores = model.predict(test_images)
labels = np.argmax(test_labels, axis=1)
idxs = ~(pred_labels == labels)
# Missing parts?

# %%

misclassified_images = test_images[idxs]
for idx in range(10000):
    if idxs[idx]:
        img = test_images[idx]
        plt.imshow(255 - img[::,], cmap = 'gray')  # Right??
        plt.xlabel('Class = ' + str(train_labels[idx]))
        plt.show()

# Missing parts?
