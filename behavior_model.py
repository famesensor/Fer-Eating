# TODO: refactor code now to function base
import keras
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Flatten, Input
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.models import Sequential

# download vgg16 weight
# https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3


def init_model_vgg16(include_top: bool, input_tensor, input_shape: tuple):
    model = VGG16(include_top=include_top, weights='imagenet',
                  input_tensor=None, input_shape=input_shape)
    return model


def setup_architechture_vgg16(model: Model):
    # don't train existing weights
    for layer in model.layers:
        layer.trainable = False

    x = Flatten()(model.output)
    prediction = Dense(2, activation='softmax')(x)
    new_model = Model(inputs=model.input, outputs=prediction)
    new_model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.Adam(),
                      metrics=['accuracy'])
    # new_model.summary()
    return new_model


def setup_architechture_vgg16_2(model: Model):
    for layers in (model.layers)[:19]:
        layers.trainable = False

    X = model.layers[-2].output
    predictions = Dense(2, activation="softmax")(X)
    new_model = Model(inputs=model.input, outputs=predictions)
    new_model.compile(loss="categorical_crossentropy",
                      optimizer=optimizers.Adam(), metrics=["accuracy"])
    # new_model.summary()
    return new_model


def train_model_vgg16(checkpoint_path: str, save_weights_path: str, model: Model,  train_data, validation_data, step_size_train, step_size_valid, epochs_train: int):
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1,
                                 save_best_only=True, save_weights_only=False, mode='auto', period=1)

    # lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
    #                                cooldown=0,
    #                                patience=5,
    #                                min_lr=0.5e-6)
    # checkpoint = ModelCheckpoint(filepath=weight_model_path,
    #                              verbose=1, save_best_only=True)

    # callbacks = [checkpoint, lr_reducer]

    early = EarlyStopping(monitor='val_acc', min_delta=0,
                          patience=40, verbose=1, mode='auto')

    callbacks = [checkpoint, early]

    start = datetime.now()
    history = model.fit_generator(train_data,
                                  steps_per_epoch=step_size_train,
                                  epochs=epochs_train, verbose=5,
                                  validation_data=validation_data,
                                  validation_steps=step_size_valid, callbacks=callbacks)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    # model.save_weights("vgg_16_behavior_1.h5") # use -> model.load_weights()
    model.save_weights(save_weights_path)
    return model, history


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
