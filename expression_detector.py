import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications import ResNet50V2, MobileNetV2
from preparation import load_image
from expression_model import setup_network

if __name__ == "__main__":
    # init value
    img_height, img_weight = 224, 224
    channels = 3  # 3 RGB
    batch_size = 32

    image_test_Path = "./data-set/behavior/test/506.jpg"
    weight_model_vgg16 = "./models/expression/vgg16/vgg16_expression.h5"
    include_top = True
    class_num = 8
    activation = "softmax"
    loss = "categorical_crossentropy"
    dict_Label = {0: "neutral", 1: "anger",
                  2: "contempt", 3: "disgust", 4: "fear", 5: "happy", 6: "sadness", 7: "surprise"}

    print("expression detection....")

    # load test image
    print("load image...")
    test_image = load_image(test_data_path=image_test_Path, size_image=(
        img_height, img_weight, channels), color_mode='rgb')
    print("load image success...")

    # load model vgg16
    print("load model vgg16...")
    vgg16 = VGG16(include_top=include_top, input_tensor=None,
                  input_shape=(img_height, img_weight, channels))
    model_vgg16 = setup_network(model=vgg16, include_top=include_top,
                                class_num=class_num, layer_num=19, activation=activation, loss=loss)
    model_vgg16.load_weights(weight_model_vgg16)
    print("load weight vgg16 success...")

    # load model vgg19

    # load model resnet

    # load model alexnet

    # load model mobilenet
