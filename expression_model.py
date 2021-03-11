from keras.applications.vgg16 import VGG16
from keras.applications import ResNet50V2


def init_model_vgg16(include_top: bool, input_tensor, input_shape: tuple):
    model = VGG16(include_top=include_top, weights='imagenet',
                  input_tensor=None, input_shape=input_shape)
    return model


def init_model_resnet50(include_top: bool, input_tensor, input_shape: tuple):
    model = ResNet50V2(include_top=include_top, weights='imagenet',
                       input_tensor=input_tensor, input_shape=input_shape)
    return model


# def init_model_mobilenet(include_top: bool, input_tensor, input_shape: tuple, minimalistic: bool, alpha: float, dropout_rate: float):
#     model =
#     return model

def init_model_alexnet():
    return
