from keras.preprocessing.image import ImageDataGenerator


def load_data_set(datagen_args: dict, data_set_path: str, size_image: tuple, batch_size: float):
    data_generator = ImageDataGenerator(**datagen_args)
    data_set = data_generator.flow_from_directory(
        directory=data_set_path, target_size=size_image, class_mode="catrgorical", batch_size=batch_size)
    return data_set
