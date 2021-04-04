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
from keras import models, layers
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dropout, Conv2D, MaxPooling2D


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


def init_model_mobilenet(include_top: bool, input_tensor, input_shape: tuple) -> Model:
    model = MobileNetV2(include_top=include_top, weights='imagenet',
                        input_tensor=None, input_shape=input_shape)
    return model


# def init_model_mobilenet_small(include_top: bool, input_tensor, input_shape: tuple, minimalistic: bool, alpha: float, dropout_rate: float):
#     return


def init_model_alexnet():
    model = models.Sequential()

    # Convolutional Layer
    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(
        4, 4), activation='relu', input_shape=(224, 224, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(5, 5),
                     strides=(1, 1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3),
                     strides=(1, 1), activation='relu', padding="same"))
    model.add(BatchNormalization())

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(1, 1),
                     strides=(1, 1), activation='relu', padding="same"))
    model.add(BatchNormalization())

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(1, 1),
                     strides=(1, 1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Fully Connected layer
    # 1st Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    # 2nd Fully Connected Layer
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    # 3rd Fully Connected Layer
    model.add(Dense(8, activation='softmax'))

    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizers.Adam(), metrics=['accuracy'])
    return model


def setup_network(model: Model, include_top: bool, class_num: int, layer_num: int, activation: str, loss: str):
    if include_top:
        for layer in model.layers[:layer_num]:
            layer.trainable = False
        x = model.layers[-2].output
        prediction = Dense(class_num, activation=activation)(x)

    new_model = Model(inputs=model.input, outputs=prediction)
    new_model.compile(loss=loss,
                      optimizer=optimizers.Adam(),
                      metrics=['accuracy'])
    return new_model


def train_model(checkpoint_path: str, save_weight_path: str, model: Model, train_data, validation_data, step_size_train, step_size_valid, epochs_train: int) -> (Model, None):
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1,
                                 save_best_only=True, save_weights_only=False, mode='auto', period=1)

    early = EarlyStopping(monitor='val_acc', min_delta=0,
                          patience=40, verbose=1, mode='auto')

    callbacks = [checkpoint, early]

    start = datetime.now()
    print("Training model in time: ", start)
    history = model.fit_generator(train_data,
                                  steps_per_epoch=step_size_train,
                                  epochs=epochs_train, verbose=5,
                                  validation_data=validation_data,
                                  validation_steps=step_size_valid, callbacks=callbacks)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    # save weight model
    model.save_weights(save_weight_path)

    return model, history


def evaluate_model_vgg16(model: Model, test_data_set):
    score = model.evaluate(test_data_set)
    return score


def plot_result_train_model(history, model_name: str):
    plt.plot(history.history["acc"])
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(model_name)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
    plt.show()
    return
