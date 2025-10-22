# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:17:21 2023

@author: IIcetiner
"""

import numpy as np

np.random.seed(1000)
import matplotlib.pyplot as plt
import keras
import os
import zipfile
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    BatchNormalization,
)
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
)
import pickle
import pandas as pd
import random
import cv2

np.random.seed(0)
from PIL import Image
import PIL
from sklearn.metrics import confusion_matrix
from PIL import Image, ImageFilter
import skimage

train_directory = "../Datasets/train/"
val_directory = "../Datasets/validation/"


def removeBackground(orijinal):
    hsv = cv2.cvtColor(orijinal, cv2.COLOR_BGR2HSV)
    image = hsv
    white_bg = 255 * np.ones_like(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    x, y, w, h = cv2.boundingRect(thresh)
    ROI = hsv[y : y + h, x : x + w]
    ROI_original = orijinal[y : y + h, x : x + w]
    tt = white_bg
    tt[y : y + h, x : x + w] = ROI
    return tt


SIZE = 64
boyut = 64
veriseti = []
etiket = []

blackspot_images = os.listdir(train_directory + "blackspot/")
for i, image_name in enumerate(blackspot_images):
    image = cv2.imread(train_directory + "blackspot/" + image_name)
    image = Image.fromarray(image, "RGB")
    image = image.resize((boyut, boyut))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = removeBackground(image)
    veriseti.append(np.array(image))
    etiket.append(0)

canker_images = os.listdir(train_directory + "canker/")
for i, image_name in enumerate(canker_images):
    # print(image_name)
    # print(i)
    image = cv2.imread(train_directory + "canker/" + image_name)
    image = Image.fromarray(image, "RGB")
    image = image.resize((boyut, boyut))
    image = np.array(image)
    # image=removeBackground(image)
    veriseti.append(np.array(image))
    etiket.append(1)

greening_images = os.listdir(train_directory + "greening/")
for i, image_name in enumerate(greening_images):
    image = cv2.imread(train_directory + "greening/" + image_name)
    image = Image.fromarray(image, "RGB")
    image = image.resize((boyut, boyut))
    image = np.array(image)
    image = removeBackground(image)
    veriseti.append(np.array(image))
    etiket.append(2)

healthy_images = os.listdir(train_directory + "healthy/")
for i, image_name in enumerate(healthy_images):
    image = cv2.imread(train_directory + "healthy/" + image_name)
    image = Image.fromarray(image, "RGB")
    image = image.resize((boyut, boyut))
    image = np.array(image)
    image = removeBackground(image)
    veriseti.append(np.array(image))
    etiket.append(3)

del blackspot_images, canker_images, greening_images, healthy_images

### verisetini bölümlendir.
## 1 verisetini eğitim ve test olarak ikiye ayırmak istiyorum.
# 2. Eğitim verisi: 80%

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

donusum = to_categorical(np.array(etiket))
X_train, X_test, y_train, y_test = train_test_split(
    veriseti, donusum, test_size=0.20, random_state=0
)

del veriseti, etiket

classes = ["blackspot", "canker", "greening", "healthy"]
batch_size = 32

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(
    12, drop_remainder=False
)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(
    12, drop_remainder=False
)

train_dataset.batch(batch_size)
test_dataset.batch(batch_size)

from tensorflow.keras import layers, datasets, models

augment1 = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

augment = ImageDataGenerator(rotation_range=25, zoom_range=0.2)

from tensorflow.keras.utils import plot_model

cnn1 = models.Sequential(
    [
        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(SIZE, SIZE, 3),
        ),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(axis=-1),
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(axis=-1),
        layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(
            256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.002)
        ),
        layers.Dense(4, activation="softmax"),
    ]
)

cnn1 = Sequential()
cnn1.add(
    Conv2D(
        filters=32, kernel_size=(3, 3), activation="relu", input_shape=(SIZE, SIZE, 3)
    )
)
cnn1.add(MaxPooling2D(pool_size=(2, 2)))
cnn1.add(BatchNormalization(axis=-1))

cnn1.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
cnn1.add(MaxPooling2D(pool_size=(2, 2)))
cnn1.add(BatchNormalization(axis=-1))

cnn1.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
cnn1.add(MaxPooling2D(pool_size=(2, 2)))

cnn1.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
cnn1.add(MaxPooling2D(pool_size=(2, 2)))

cnn1.add(Flatten())
cnn1.add(Dropout(0.4))
cnn1.add(
    Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.002))
)
cnn1.add(Dense(4, activation="softmax"))
