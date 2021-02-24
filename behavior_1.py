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
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential

# download vgg16 weight
# https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

# init value
img_height, img_weight = 224, 244
channels = 3  # 3 RGB
train_data_path = "./data-set/behavior/train"
validation_data_path = "./data-set/behavior/validate"
test_data_path = "./data-set/behavior/test"
weight_model_path = "./model/behavior/vgg16_weights"
batch_size = 32
Epochs = 18
dict_label = {0: "eat", 1: "noeat"}
datagen_args = dict(
    rotation_range=20,
    rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

trian_datagen = ImageDataGenerator(**datagen_args)
train_data = trian_datagen.flow_from_directory(directory=train_data_path, target_size=(
    img_height, img_weight), subset='training', class_mode="categorical", interpolation="lanczos", batch_size=batch_size, shuffle=True)
validation_data = trian_datagen.flow_from_directory(directory=validation_data_path, target_size=(
    img_height, img_weight), subset='validation', class_mode="categorical", interpolation="lanczos", batch_size=batch_size, shuffle=True)

# test_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
# test_data = test_datagen.flow_from_directory(
#     directory=test_data_path, target_size=(img_height, img_weight), batch_size=batch_size, class_mode='categorical')

STEP_SIZE_TRAIN = train_data.n//train_data.batch_size
STEP_SIZE_VALID = validation_data.n//validation_data.batch_size
# STEP_SIZE_TEST = test_data.n//test_data.batch_size

# Model VGG16_1
vggmodel = VGG16(include_top=False, weights='imagenet',
                 input_tensor=None, input_shape=(img_height, img_weight, 3))

# # don't train existing weights
for layer in vggmodel.layers:
    layer.trainable = False

x = Flatten()(vggmodel.output)
prediction = Dense(2, activation='softmax')(x)
model = Model(inputs=vggmodel.input, outputs=prediction)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(),
              metrics=['accuracy'])
model.summary()


# lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
#                                cooldown=0,
#                                patience=5,
#                                min_lr=0.5e-6)
# checkpoint = ModelCheckpoint(filepath=weight_model_path,
#                              verbose=1, save_best_only=True)

# callbacks = [checkpoint, lr_reducer]

checkpoint = ModelCheckpoint(weight_model_path, monitor='val_acc', verbose=1,
                             save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0,
                      patience=40, verbose=1, mode='auto')

callbacks = [checkpoint, early]

start = datetime.now()
history = model.fit_generator(train_data,
                              steps_per_epoch=STEP_SIZE_TRAIN,
                              epochs=Epochs, verbose=5,
                              validation_data=validation_data,
                              validation_steps=STEP_SIZE_VALID, callbacks=callbacks)

duration = datetime.now() - start
print("Training completed in time: ", duration)
model.save_weights("vgg_16_behavior_1.h5")

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
