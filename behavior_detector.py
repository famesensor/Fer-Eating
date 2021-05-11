import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from behavior.behavior_model import setup_architechture_vgg16, setup_architechture_vgg16_2

# init value
img_height, img_weight = 224, 224
channels = 3  # 3 RGB
batch_size = 32

image_test_Path = "./data-set/behavior/test/506.jpg"
# images_Test_Path = "./data-set/behavior/test/"
weight_model_path_one = "./models/behavior/vgg16_behavior_1.h5"
weight_model_path_two = "./models/behavior/vgg16_behavior_2.h5"
dict_Label = {0: "eat", 1: "noeat"}

# test_datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True)
# test_data = test_datagen.flow_from_directory(
#     directory=images_Test_Path, target_size=(img_height, img_weight), batch_size=batch_size, class_mode='categorical')

# STEP_SIZE_TEST = test_data.n//test_data.batch_size

print("load image....")
test_image = image.load_img(
    image_test_Path, color_mode='rgb', target_size=(img_height, img_weight))  # color_mode = rgb or grayscale

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

print("load image success....")

# plt.imshow(test_image)

# load model
print("load model....")
# model = load_model(weight_model_path_one)
# init model one
vggmodel_one = VGG16(include_top=False, input_tensor=None,
                     input_shape=(img_height, img_weight, channels))
model_one = setup_architechture_vgg16(vggmodel_one)
model_one.load_weights(weight_model_path_one)

# init model two
vggmodel_two = VGG16(include_top=True, input_tensor=None,
                     input_shape=(img_height, img_weight, channels))
model_two = setup_architechture_vgg16_2(vggmodel_two)
model_two.load_weights(weight_model_path_two)

print("load weight success....")

# predict
output = model_one.predict(test_image)
output_two = model_two.predict(test_image)

# check output
# if output[0][0] > output[0][1]:
#     print("eat")
# else:
#     print("noeat")
res_one = np.argmax(output)
res_two = np.argmax(output_two)
print("The predicted output from model 1 :", dict_Label[res_one], "\nsucscess")
print("The predicted output from model 1 :", dict_Label[res_two], "\nsucscess")
