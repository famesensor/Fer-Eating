import numpy as np
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image


def load_data_set(datagen_args: dict, data_set_path: str, size_image: tuple, batch_size: float):
    data_generator = ImageDataGenerator(**datagen_args)
    data_set = data_generator.flow_from_directory(
        directory=data_set_path, target_size=size_image, class_mode="categorical", batch_size=batch_size)
    return data_set


def load_image(test_data_path: str, size_image: tuple, color_mode: str):
    test_image = image.load_img(
        test_data_path, color_mode=color_mode, target_size=size_image)

    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    return test_image
