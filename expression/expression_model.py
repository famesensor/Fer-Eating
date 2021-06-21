import keras
import numpy as np

from datetime import datetime
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Flatten, Input, GlobalAveragePooling2D
from keras import optimizers
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications import ResNet50V2, MobileNetV2
from keras.models import Model
from keras import models, layers
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dropout, Conv2D, MaxPooling2D


def init_model_train_expression(types: str, include_top: bool, img_height: int, img_width: int, channels: int, class_num: int, layer_num: int, activation: str, loss: str, dropout=0.2) -> Model:
    model = {
        "vgg16": VGG16(include_top=include_top, input_tensor=None, weights='imagenet',
                       input_shape=(img_height, img_width, channels)),
        "vgg19": VGG19(include_top=include_top, input_tensor=None, weights='imagenet',
                       input_shape=(img_height, img_width, channels)),
        "resnet": ResNet50V2(include_top=include_top, input_tensor=None, weights='imagenet',
                             input_shape=(img_height, img_width, channels)),
        "mobilenet": MobileNetV2(include_top=include_top, input_tensor=None, weights='imagenet', input_shape=(
            img_height, img_width, channels)),
    }[types]

    model = setup_network(model=model, include_top=include_top,
                          class_num=class_num, layer_num=layer_num, activation=activation, loss=loss, types=types, dropout=dropout)
    print("[INFO]: init model behavior {}...".format(types))
    return model


def setup_network(model: Model, include_top: bool, class_num: int, layer_num: int, activation: str, loss: str, types: str,  dropout: float) -> Model:
    if include_top:
        for layer in model.layers[:layer_num]:
            layer.trainable = False
        x = model.layers[-2].output
        prediction = Dense(class_num, activation=activation)(x)
    else:
        for layer in model.layers:
            layer.trainable = False
        if types in ["vgg16", "vgg19", "resnet"]:
            x = Flatten(name='flatten')(model.output)
            x = Dense(units=512, activation='relu', name='fc1')(x)
            x = Dense(units=256, activation='relu', name='fc2')(x)
            x = Dropout(dropout, name='dropout_1')(x)
            x = Dense(units=128, activation='relu', name='fc3')(x)
            x = Dropout(dropout, name='dropout_2')(x)
            prediction = Dense(class_num, activation=activation)(x)
        if types == "mobilenet":
            x = GlobalAveragePooling2D()(model.output)
            x = Dense(units=512, activation='relu', name='fc1')(x)
            x = Dense(units=256, activation='relu', name='fc2')(x)
            x = Dropout(dropout, name='dropout_1')(x)
            x = Dense(units=128, activation='relu', name='fc3')(x)
            x = Dropout(dropout, name='dropout_2')(x)
            prediction = Dense(class_num, activation=activation)(x)

    new_model = Model(inputs=model.input, outputs=prediction)
    new_model.compile(loss=loss,
                      optimizer=optimizers.Adam(),
                      metrics=['accuracy'])  # learning_rate=1e-5
    return new_model


def train_model(checkpoint_path: str, save_weight_path: str, model: Model, train_data, validation_data, step_size_train, step_size_valid, epochs_train: int) -> (Model, None):
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1,
                                 save_best_only=True, save_weights_only=False, mode='auto', period=1)

    early = EarlyStopping(monitor='val_accuracy', min_delta=0,
                          patience=40, verbose=1, mode='auto')

    callbacks = [checkpoint, early]

    start = datetime.now()
    print("Training model in time: ", start)
    history = model.fit(train_data,
                        steps_per_epoch=step_size_train,
                        epochs=epochs_train, verbose=2,
                        validation_data=validation_data,
                        validation_steps=step_size_valid, callbacks=callbacks)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    # save weight model
    model.save_weights(save_weight_path)

    return model, history


def evaluate_model(model: Model, test_data_set):
    score = model.evaluate(test_data_set)
    return score


def init_model_expression(weight_path: str, types: str, include_top: bool, img_height: int, img_weight: int, channels: int, class_num: int, layer_num: int, activation: str, loss: str) -> Model:
    model = {
        "vgg16": VGG16(include_top=include_top, input_tensor=None,
                       input_shape=(img_height, img_weight, channels)),
        "vgg19": VGG19(include_top=include_top, input_tensor=None,
                       input_shape=(img_height, img_weight, channels)),
        "resnet": ResNet50V2(include_top=include_top, input_tensor=None,
                             input_shape=(img_height, img_weight, channels)),
        "mobilenet": MobileNetV2(include_top=include_top, input_tensor=None, input_shape=(
            img_height, img_weight, channels)),
    }[types]

    model = setup_network(model=model, include_top=include_top,
                          class_num=class_num, layer_num=layer_num, activation=activation, loss=loss, types=types, dropout=0.2)
    model.load_weights(weight_path)
    print("[INFO]: init model expression {}...".format(types))

    return model
