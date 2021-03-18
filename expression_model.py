import keras
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Flatten, Input
from keras import optimizers
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications import ResNet50V2, MobileNetV2
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


def setup_network(model: Model, include_top: bool, class_num: int, layer_num: int, activation: str, loss: str):
    if include_top:
        for layer in model.layers[:layer_num]:
            layer.trainable = False
        x = model.layers[-2].outputs
        prediction = Dense(class_num, activation=activation)(x)

    new_model = Model(inputs=model.input, outputs=prediction)
    new_model.compile(loss=loss,
                      optimizer=optimizers.Adam(),
                      metrics=['accuracy'])
    return new_model


# def setup_network_vgg16(model: Model, include_top: bool, class_num: int) -> Model:
#     if include_top:
#         for layer in model.layers[:19]:
#             layer.trainable = False
#         x = model.layers[-2].outputs
#         prediction = Dense(class_num, activation='softmax')(x)
#     # else:
#     #     for layer in model.layers:
#     #         layer.trainable = False

#     new_model = Model(inputs=model.input, outputs=prediction)
#     new_model.compile(loss='categorical_crossentropy',
#                       optimizer=optimizers.Adam(),
#                       metrics=['accuracy'])
#     return new_model


# def setup_network_vgg19(model: Model, include_top: bool, class_num: int) -> Model:
#     if include_top:
#         for layer in model.layers[:22]:
#             layer.trainable = False
#         x = model.layers[-2].outputs
#         prediction = Dense(class_num, activation='softmax')(x)

#     new_model = Model(input=model.input, output=prediction)
#     new_model.compile(loss='categorical_crossentropy',
#                       optimizer=optimizers.Adam(),
#                       metrics=['accuracy'])
#     return new_model


# def setup_network_resnet(model: Model, include_top: bool, class_num: int) -> Model:
#     if include_top:
#         for layer in model.layers[:190]:
#             layer.trainable = False
#         x = model.layers[-2].outputs
#         prediction = Dense(class_num, activation='softmax')(x)

#     new_model = Model(input=model.input, output=prediction)
#     new_model.compile(loss='categorical_crossentropy',
#                       optimizer=optimizers.Adam(),
#                       metrics=['accuracy'])
#     return new_model


# def setup_network_mobilenet(model: Model, include_top: bool, class_num: int) -> Model:
#     if include_top:
#         for layer in model.layers[:154]:
#             layer.trainable = False
#         x = model.layers[-2].outputs
#         prediction = Dense(class_num, activation='softmax')(x)

#     new_model = Model(input=model.input, output=prediction)
#     new_model.compile(loss='categorical_crossentropy',
#                       optimizer=optimizers.Adam(),
#                       metrics=['accuracy'])
#     return new_model


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
