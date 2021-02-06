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

# init value
img_height, img_weight = 224, 244
channels = 3  # 3 RGB
train_data_path = "./data-set/behavior/train"
validation_data_path = "./data-set/behavior/validate"
batch_size = 512
classes_name = "binary"
# classes_dict = ["eat", "noeat"]

# TODO: Add batch-size and import test dataset
trian_datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True)
train_data = trian_datagen.flow_from_directory(directory=train_data_path, target_size=(
    img_height, img_weight), subset="training", class_mode="categorical")
validation_data = trian_datagen.flow_from_directory(directory=validation_data_path, target_size=(
    img_height, img_weight), subset="validation", class_mode="categorical")

# # Model VGG16_1
vggmodel1 = VGG16(include_top=True, weights='imagenet',
                  input_tensor=None, input_shape=None)

# don't train existing weights
for layer in vggmodel1.layers:
    layer.trainable = False

x = Flatten()(vggmodel1.output)
prediction = Dense(2, activation='softmax')(x)
model = Model(inputs=vggmodel1.input, outputs=prediction)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(),
              metrics=['accuracy'])
model.summary()
