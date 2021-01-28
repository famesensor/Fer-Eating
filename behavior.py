import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# init value
img_height, img_weight = 224, 244
channels = 3  # 3 RGB
train_data_path = "/data-set/behavior/train"
validation_data_path = "/data-set/behavior/validate"
batch_size = 512

# TODO: Add batch-size and import test dataset
trian_datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True)
train_data = trian_datagen.flow_from_directory(directory=train_data_path, target_size=(
    img_height, img_weight), subset="training", class_mode="categorical")
validation_data = trian_datagen.flow_from_directory(directory=validation_data_path, target_size=(
    img_height, img_weight), subset="validation", class_mode="categorical")
