import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications import ResNet50V2
from keras.applications import MobileNetV2
from keras.models import Model


def init_model_vgg16(include_top: bool, input_tensor, input_shape: tuple) -> Model:
    model = VGG16(include_top=include_top, weights='imagenet',
                  input_tensor=None, input_shape=input_shape)
    return model


def init_model_vgg19(include_top: bool, input_tensor, input_shape: tuple) -> Model:
    model = VGG19(include_top=include_top, weights='imagenet',
                  input_tensor=None, input_shape=input_shape)
    return model


def init_model_resnet50v2(include_top: bool, input_tensor, input_shape: tuple) -> Model:
    model = ResNet50V2(include_top=include_top, weights='imagenet',
                       input_tensor=None, input_shape=input_shape)
    return model


def init_model_mobilenet(include_top: bool, input_tensor, input_shape: tuple, minimalistic: bool, alpha: float, dropout_rate: float) -> Model:
    model = MobileNetV2(include_top=include_top, weights='imagenet',
                        input_tensor=None, input_shape=input_shape)
    return model


# def init_model_mobilenet_small(include_top: bool, input_tensor, input_shape: tuple, minimalistic: bool, alpha: float, dropout_rate: float):
#     return


def init_model_alexnet():
    return


def setup_network_vgg16(model: Model) -> Model:
    return


def setup_network_vgg19(model: Model) -> Model:
    return


def setup_network_resnet(model: Model) -> Model:
    return


def setup_network_mobilenet(model: Model) -> Model:
    return


def train_model(model: Model) -> (Model, None):
    return


def evaluate_model_vgg16(model: Model, test_data_set):
    score = model.evaluate(test_data_set)
    return score


def plot_result_train_model(history):
    plt.plot(history.history["acc"])
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
    plt.show()
    return
