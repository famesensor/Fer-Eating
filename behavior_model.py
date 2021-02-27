# TODO: refactor code now to function base
import keras
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Flatten
from keras import optimizers
from keras.models import Sequential
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
# from keras.preprocessing.image import ImageDataGenerator
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


def train_model_vgg16(checkpoint_path: str, save_weights_path: str, model: Model,  train_data, validation_data, step_size_train, step_size_valid, epochs):
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1,
                                 save_best_only=True, save_weights_only=False, mode='auto', period=1)

    early = EarlyStopping(monitor='val_acc', min_delta=0,
                          patience=40, verbose=1, mode='auto')

    callbacks = [checkpoint, early]

    start = datetime.now()
    history = model.fit_generator(train_data,
                                  steps_per_epoch=step_size_train,
                                  epochs=epochs, verbose=5,
                                  validation_data=validation_data,
                                  validation_steps=step_size_valid, callbacks=callbacks)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    # model.save_weights("vgg_16_behavior_1.h5")
    model.save_weights(save_weights_path)
    return model

# lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
#                                cooldown=0,
#                                patience=5,
#                                min_lr=0.5e-6)
# checkpoint = ModelCheckpoint(filepath=weight_model_path,
#                              verbose=1, save_best_only=True)

# callbacks = [checkpoint, lr_reducer]


# # evalute model
# score = model.evaluate(test_data)
# print('Test Loss:', score[0])
# print('Test accuracy:', score[1])

# plot result train model
plt.plot(history.history["acc"])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
plt.show()

# Test model with image
test_image = image.load_img(
    test_data_path, color_mode='rgb', target_size=(224, 224))

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)

res = np.argmax(result)
print("The predicted output is :", dict_label[res])
