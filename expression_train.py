import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.image import ImageDataGenerator

from expression.expression_model import init_model_train_expression, train_model, plot_result_train_model


if __name__ == "__main__":
    CATEGORIES = ["anger", "contempt", "disgust", "fear",
                  "happy", "neutral", "sadness", "surprise"]

    iteration = 1
    batch_size = 32
    img_size = (224, 224)
    train_dir = './dataset/expression/train'
    train_datagen_args = dict(
        rotation_range=20,
        rescale=1./255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    for category in CATEGORIES:
        print('{} {} images'.format(category, len(
            os.listdir(os.path.join(train_dir, category)))))

    train = []
    for category_id, category in enumerate(CATEGORIES):
        for file in os.listdir(os.path.join(train_dir, category)):
            train.append(
                ['train/{}/{}'.format(category, file), category_id, category])
    train = pd.DataFrame(train, columns=['file', 'category_id', 'category'])

    # k-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True)

    # image generate
    data_generator = ImageDataGenerator(**train_datagen_args)
    data_generator_valid = ImageDataGenerator(rescale=1./255)

    t = train.category_id
    for train_index, test_index in skf.split(np.zeros(len(t)), t):
        print("======================================")
        print("Iteration = ", iteration)

        iteration = iteration + 1

        train_set = train.iloc[train_index]
        test_set = train.iloc[test_index]

        print("======================================")

        train_generator = data_generator.flow_from_dataframe(dataframe=train_set,
                                                             directory='./expression/',
                                                             x_col="file",
                                                             y_col="category",
                                                             batch_size=batch_size,
                                                             shuffle=True,
                                                             class_mode="categorical",
                                                             target_size=img_size)
        valid_generator = data_generator_valid.flow_from_dataframe(dataframe=test_set,
                                                                   directory="./expression/",
                                                                   x_col="file",
                                                                   y_col="category",
                                                                   batch_size=batch_size,
                                                                   shuffle=False,
                                                                   class_mode="categorical",
                                                                   target_size=img_size)

        STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
        STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size
