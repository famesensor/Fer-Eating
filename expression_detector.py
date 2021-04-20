import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications import ResNet50V2, MobileNetV2
from preparation.preparation import load_image
from expression.expression_model import setup_network
from detection.object_detector import face_detect

if __name__ == "__main__":
    # init value
    img_height, img_weight = 224, 224
    channels = 3  # 3 RGB
    batch_size = 32

    image_test_Path = "./data-set/behavior/test/506.jpg"
    weight_model_vgg16 = "./models/expression/vgg16/vgg16_expression.h5"
    weight_model_vgg19 = "./models/expression/vgg19/vgg19_expression.h5"
    weight_model_resnet = "./models/expression/resnet/resnet_expression.h5"
    weight_model_alexnet = "./models/expression/alexnet/alexnet_expression.h5"
    weight_model_mobilenet = "./models/expression/mobilenet/mobilenet_expression.h5"
    include_top = True
    class_num = 8
    activation = "softmax"
    loss = "categorical_crossentropy"
    dict_exppression = {0: "anger", 1: "contempt", 2: "disgust",
                        3: "fear", 4: "happay", 5: "neutral", 6: "sadness", 7: "surprise"}

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
    vgg19 = VGG19(include_top=include_top, input_tensor=None,
                  input_shape=(img_height, img_weight, channels))
    model_vgg19 = setup_network(model=vgg19, include_top=include_top,
                                class_num=class_num, layer_num=22, activation=activation, loss=loss)
    model_vgg19.load_weights(weight_model_vgg19)

    # load model resnet
    resnet = ResNet50V2(include_top=include_top, input_tensor=None,
                        input_shape=(img_height, img_weight, channels))
    model_resnet = setup_network(model=resnet, include_top=include_top,
                                 class_num=class_num, layer_num=190, activation=activation, loss=loss)
    model_resnet.load_weights(weight_model_resnet)

    # load model alexnet

    # load model mobilenet
    mobilenet = MobileNetV2(include_top=include_top, input_tensor=None, input_shape=(
        img_height, img_weight, channels))
    model_mobile = setup_network(model=mobilenet, include_top=include_top,
                                 class_num=class_num, layer_num=154, activation=activation, loss=loss)
    model_mobile.load_weights(weight_model_mobilenet)

test_image = face_detect(image=test_image)
test_image = np.expand_dims(test_image, axis=0)

# predict
print("predict...")
output_vgg16 = model_vgg16.predict(test_image)
output_vgg19 = model_vgg19.predict(test_image)
output_resnet = model_resnet.predict(test_image)
output_mobile = model_mobile.predict(test_image)

# convert result
res_vgg16 = np.argmax(output_vgg16)
res_vgg19 = np.argmax(output_vgg19)
res_resnet = np.argmax(output_resnet)
res_mobile = np.argmax(output_mobile)

print("The predicted output from model vgg16 : ",
      dict_exppression[res_vgg16], "success \n")
print("The predicted output from model vgg19 : ",
      dict_exppression[res_vgg19], "success \n")
print("The predicted output from model resnet : ",
      dict_exppression[res_resnet], "success \n")
print("The predicted output from model mobile : ",
      dict_exppression[res_mobile], "success \n")
