# TODO: refactor code now to function base

import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import load_model

# init value
img_height, img_weight = 224, 224
batch_size = 32

image_test_Path = "./data-set/behavior/test/499.jpg"
# images_Test_Path = "./data-set/behavior/test/"
weight_model_path_one = "./models/behavior/vgg16_behavior_1.h5"
weight_model_path_two = "./models/behavior/vgg16_behavior_2.h5"
dict_Label = {0: "eat", 1: "noeat"}

# test_datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True)
# test_data = test_datagen.flow_from_directory(
#     directory=images_Test_Path, target_size=(img_height, img_weight), batch_size=batch_size, class_mode='categorical')

# STEP_SIZE_TEST = test_data.n//test_data.batch_size

test_image = image.load_img(
    image_test_Path, color_mode='grayscale', target_size=(img_height, img_weight))  # color_mode = rgb or grayscale

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# plt.imshow(test_image)

# load model
model = load_model(weight_model_path_one)

# predict
output = model.predict(test_image)

# check output
if output[0][0] > output[0][1]:
    print("eat")
else:
    print("noeat")

res = np.argmax(output)
print("The predicted output is :", dict_Label[res])
