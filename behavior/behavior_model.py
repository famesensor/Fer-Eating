import keras
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Flatten, Input, Dropout, GlobalAveragePooling2D
from keras import optimizers
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications import ResNet50V2, MobileNetV2
from keras.models import Sequential


def init_model_train_behavior(types: str, include_top: bool, img_height: int, img_weight: int, channels: int, class_num: int, layer_num: int, activation: str, loss: str, dropout=0.2) -> Model:
    model = {
        "vgg16": VGG16(include_top=include_top, input_tensor=None, weights='imagenet',
                       input_shape=(img_height, img_weight, channels)),
        "vgg19": VGG19(include_top=include_top, input_tensor=None, weights='imagenet',
                       input_shape=(img_height, img_weight, channels)),
        "resnet": ResNet50V2(include_top=include_top, input_tensor=None, weights='imagenet',
                             input_shape=(img_height, img_weight, channels)),
        "mobilenet": MobileNetV2(include_top=include_top, input_tensor=None, weights='imagenet', input_shape=(
            img_height, img_weight, channels)),
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
        if types in ["vgg16", "vgg19"]:
            x = Flatten()(model.output)
            x = Dense(units=4096, activation='relu')(x)
            x = Dropout(dropout)(x)
            x = Dense(units=4096, activation='relu')(x)
            x = Dropout(dropout)(x)
            prediction = Dense(class_num, activation=activation)(x)
        if types == "resnet":
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
                      metrics=['accuracy'])
    return new_model


def train_model(checkpoint_path: str, save_weights_path: str, model: Model,  train_data, validation_data, step_size_train, step_size_valid, epochs_train: int):
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1,
                                 save_best_only=True, save_weights_only=False, mode='auto', period=1)

    early = EarlyStopping(monitor='val_acc', min_delta=0,
                          patience=40, verbose=1, mode='auto')

    callbacks = [checkpoint, early]

    start = datetime.now()
    history = model.fit_generator(train_data,
                                  steps_per_epoch=step_size_train,
                                  epochs=epochs_train, verbose=5,
                                  validation_data=validation_data,
                                  validation_steps=step_size_valid, callbacks=callbacks, shuffle=True)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    # model.save_weights("vgg_16_behavior_1.h5") # use -> model.load_weights()
    model.save_weights(save_weights_path)
    return model, history


def init_model_behavior(weight_path: str, types: str, include_top: bool, img_height: int, img_weight: int, channels: int, class_num: int, layer_num: int, activation: str, loss: str) -> Model:
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
    print("[INFO]: init model behavior {}...".format(types))
    return model
