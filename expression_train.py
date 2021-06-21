import os
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.image import ImageDataGenerator

from expression.expression_model import init_model_train_expression, train_model
from plot.plot import plot_result_train_model

if __name__ == "__main__":
    CATEGORIES = ["anger", "contempt", "disgust", "fear",
                  "happy", "neutral", "sadness", "surprise"]

    iteration = 1
    include_top = False
    Epochs = 1
    preTrainType = "mobilenet"  # vgg19, resnet, mobilenet
    layer_num = 19  # {vgg16: 19, vgg19: 22, resnet: 190, mobilenet: 154}
    dropout = 0.2
    class_num = 8
    batch_size = 32
    img_h, img_w = 224, 224
    channels = 3
    img_size = (img_h, img_w)
    activation = "softmax"
    loss = "categorical_crossentropy"
    train_dir = './dataset/expression/'
    train_datagen_args = dict(rescale=1./255)
    VALIDATION_ACCURACY = []
    VALIDAITON_LOSS = []
    save_weight_dir = "./models/expression/"+preTrainType+"/"+preTrainType

    # sum all to dataframe
    CATEGORIES = ["anger", "contempt", "disgust", "fear",
                  "happy", "neutral", "sadness", "surprise"]
    for category in CATEGORIES:
        print('{} {} images'.format(category, len(
            os.listdir(os.path.join(train_dir+'train/', category)))))

    train = []
    for category_id, category in enumerate(CATEGORIES):
        for file in os.listdir(os.path.join(train_dir+'train/', category)):
            if file != ".DS_Store":
                train.append(
                    ['train/{}/{}'.format(category, file), category_id, category])
    train = pd.DataFrame(train, columns=['file', 'category_id', 'category'])

    # k-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True)

    # image generate
    data_generator = ImageDataGenerator(**train_datagen_args)
    data_generator_valid = ImageDataGenerator(rescale=1./255)

    t = train.category_id

    # file result...
    file_result = open(save_weight_dir+"_result.txt", "w")
    file_result.write("Model name: "+preTrainType+"\n")

    for train_index, test_index in skf.split(np.zeros(len(t)), t):
        print("======================================")
        print("Iteration = ", iteration)

        train_set = train.iloc[train_index]
        test_set = train.iloc[test_index]

        print("======================================")

        train_generator = data_generator.flow_from_dataframe(dataframe=train_set,
                                                             directory=train_dir,
                                                             x_col="file",
                                                             y_col="category",
                                                             batch_size=batch_size,
                                                             shuffle=True,
                                                             class_mode="categorical",
                                                             target_size=img_size)
        valid_generator = data_generator_valid.flow_from_dataframe(dataframe=test_set,
                                                                   directory=train_dir,
                                                                   x_col="file",
                                                                   y_col="category",
                                                                   batch_size=batch_size,
                                                                   shuffle=False,
                                                                   class_mode="categorical",
                                                                   target_size=img_size)

        STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
        STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size

        # init model
        model = init_model_train_expression(types=preTrainType, include_top=include_top, img_height=img_h,
                                            img_width=img_w, channels=channels, class_num=class_num, layer_num=layer_num, activation=activation, loss=loss, dropout=dropout)

        # train model
        model, history = train_model(checkpoint_path=save_weight_dir+"_"+str(iteration)+"_best.h5", save_weight_path=save_weight_dir+"_"+str(iteration)+"_last.h5",
                                     model=model, train_data=train_generator, validation_data=valid_generator, step_size_train=STEP_SIZE_TRAIN, step_size_valid=STEP_SIZE_VALID, epochs_train=Epochs)

        # evaluate
        model.load_weights(save_weight_dir+"_"+str(iteration)+"_last.h5")
        results = model.evaluate(valid_generator)
        print("Accuracy: %.2f%%" % (results[1]*100))
        results = dict(zip(model.metrics_names, results))
        VALIDATION_ACCURACY.append(results['accuracy'])
        VALIDAITON_LOSS.append(results['loss'])
        file_result.write(
            "Iteration: %d -> Accuracy: %s, Loss: %s \n" % (iteration, str(results['accuracy']), str(results['loss'])))

        # plot result
        plot_result_train_model(history=history, model_name=preTrainType)

        iteration = iteration + 1

        # clear session
        keras.backend.clear_session()

    accuracy = np.mean(VALIDATION_ACCURACY)
    std = np.std(VALIDATION_ACCURACY)
    print("Accuracy: %.2f%% (+/- %.2f%%)" % (accuracy, std))
    file_result.write("Accuracy avg: %.2f%% (+/- %.2f%%)" % (accuracy, std))
    file_result.close()
