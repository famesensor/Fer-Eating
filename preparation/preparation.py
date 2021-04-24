import cv2
import numpy as np
import os
import random
import shutil

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image


def load_data_set(datagen_args: dict, data_set_path: str, size_image: tuple, batch_size: float):
    data_generator = ImageDataGenerator(**datagen_args)
    data_set = data_generator.flow_from_directory(
        directory=data_set_path, target_size=size_image, class_mode="categorical", batch_size=batch_size)
    print("[INFO]: load dataset...")
    return data_set


def load_image(test_data_path: str, size_image: tuple, color_mode: str):
    if len(size_image) != 0:
        test_image = image.load_img(
            test_data_path, color_mode=color_mode, target_size=size_image)
    else:
        test_image = image.load_img(
            test_data_path, color_mode=color_mode)

    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    print("[INFO]: load image...")
    return test_image


def load_vdo(vdo_path: str) -> cv2.VideoCapture:
    print("[INFO]: load vdo file...")
    return cv2.VideoCapture(vdo_path)


def resize_image(image: list, size_image: tuple) -> list:
    return cv2.resize(src=image, dsize=size_image)


def normalize_image(image: list) -> list:
    return image*(1.0/255)


def random_dataset_behavior() -> None:
    eatSource = '../dataset/behavior/train/eat'
    eatDest = '../dataset/behavior/validate/eat'
    eatFiles = os.listdir(eatSource)
    eatNumberOfFiles = 700

    noEatSource = '../dataset/behavior/train/noeat'
    noEatDest = '../dataset/behavior/validate/noeat'
    noEatFiles = os.listdir(noEatSource)
    noEatNumberOfFiles = 1200

    for file_name in random.sample(eatFiles, eatNumberOfFiles):
        shutil.move(os.path.join(eatSource, file_name), eatDest)

    for file_name in random.sample(noEatFiles, noEatNumberOfFiles):
        shutil.move(os.path.join(noEatSource, file_name), noEatDest)
    return


def random_dataset_expression() -> None:
    folder_list = [
        "neutral",
        "anger",
        "contempt",
        "disgust",
        "fear",
        "happy",
        "sadness",
        "surprise"
    ]

    # init number of files for validation
    neutral = 883
    anger = 162
    contempt = 65
    disgust = 212
    fear = 90
    happy = 248
    sadness = 101
    surprise = 299

    for folder_name in folder_list:
        source = '../dataset/expression/train/' + folder_name
        destination = '../dataset/expression/validate/' + folder_name
        files = os.listdir(source)
        number_of_files = eval(folder_name)

        for file_name in random.sample(files, number_of_files):
            shutil.move(os.path.join(source, file_name), destination)
    return
