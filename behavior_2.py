from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime
import keras
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Flatten
from keras import optimizers
from keras.models import Sequential
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np

# download vgg16 weight
# https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

# init value
img_height, img_weight = 224, 244
channels = 3  # 3 RGB
train_data_path = "./data-set/behavior/train"
validation_data_path = "./data-set/behavior/validate"
weight_model_path = "./model/behavior/vgg16_weights"
batch_size = 32
Epochs = 18
# classes_name = "binary"
# classes_dict = ["eat", "noeat"]

# TODO: import test dataset
trian_datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True)
train_data = trian_datagen.flow_from_directory(directory=train_data_path, target_size=(
    img_height, img_weight), subset="training", class_mode="categorical", batch_size=batch_size)
validation_data = trian_datagen.flow_from_directory(directory=validation_data_path, target_size=(
    img_height, img_weight), subset="validation", class_mode="categorical", batch_size=batch_size)

STEP_SIZE_TRAIN = train_data.n//train_data.batch_size
STEP_SIZE_VALID = validation_data.n//validation_data.batch_size
# STEP_SIZE_TEST = test_set.n//test_set.batch_size

# # Model VGG16_2
vggmodel = VGG16(include_top=True, weights='imagenet',
                 input_tensor=None, input_shape=None)

for layers in (vggmodel.layers)[:19]:
    # print(layers)
    layers.trainable = False

X = vggmodel.layers[-2].output
predictions = Dense(2, activation="softmax")(X)
model = Model(inputs=vggmodel.input, outputs=predictions)
model.compile(loss="categorical_crossentropy",
              optimizer=optimizers.Adam(), metrics=["accuracy"])
# model.summary()

checkpoint = ModelCheckpoint(weight_model_path, monitor='val_acc', verbose=1,
                             save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0,
                      patience=40, verbose=1, mode='auto')

callbacks = [checkpoint, early]

start = datetime.now()

history = model.fit_generator(train_data, steps_per_epoch=STEP_SIZE_TRAIN, epochs=Epochs, verbose=5,
                              validation_data=validation_data, validation_steps=STEP_SIZE_VALID, callbacks=callbacks)

duration = datetime.now() - start
print("Training completed in time: ", duration)
model.save_weights("vgg_16_behavior_2.5")

# evalute model
# score = model.evaluate(test_set)
# print('Test Loss:', score[0])
# print('Test accuracy:', score[1]
