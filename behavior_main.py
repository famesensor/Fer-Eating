import numpy as np

from preparation import load_data_set, load_image
from behavior_model import init_model_vgg16, setup_architechture_vgg16, setup_architechture_vgg16_2, train_model_vgg16, plot_result_train_model

if __name__ == "__main__":
    # init value
    img_height, img_weight = 224, 224
    channels = 3  # 3 RGB
    train_data_path = "./data-set/behavior/train"
    validation_data_path = "./data-set/behavior/validate"
    test_data_path = "./data-set/behavior/test"
    test_one_path = "./data-set/behavior/test/test.jpg"
    check_point_path = "./model/behavior/"
    save_weight_one_path = "./models/behavior/vgg16_behavior_1.h5"
    save_weight_two_path = "./models/behavior/vgg16_behavior_2.h5"
    batch_size = 32
    Epochs = 18
    dict_label = {0: "eat", 1: "noeat"}
    train_datagen_args = dict(
        rotation_range=20,
        rescale=1./255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    test_datagen_args = dict(
        rescale=1./255
    )

    # load data set train,test and validate
    train_data = load_data_set(
        train_datagen_args, train_data_path, (img_height, img_weight), batch_size)
    validation_data = load_data_set(
        train_datagen_args, validation_data_path, (img_height, img_weight), batch_size)
    test_data = load_data_set(
        test_datagen_args, test_data_path, (img_height, img_weight), batch_size)

    STEP_SIZE_TRAIN = train_data.n//train_data.batch_size
    STEP_SIZE_VALID = validation_data.n//validation_data.batch_size
    STEP_SIZE_TEST = test_data.n // test_data.batch_size

    # init model vgg 1
    model_1 = init_model_vgg16(
        include_top=False, input_tensor=None, input_shape=(img_height, img_weight, channels))

    model_1 = setup_architechture_vgg16(model_1)
    model_train_1, history_1 = train_model_vgg16(checkpoint_path=check_point_path, save_weights_path=save_weight_one_path, model=model_1,
                                                 train_data=train_data, validation_data=validation_data, step_size_train=STEP_SIZE_TRAIN, step_size_valid=STEP_SIZE_VALID, epochs=Epochs)
    plot_result_train_model(history_1)

    # init model vgg 2
    model_2 = init_model_vgg16(
        include_top=True, input_tensor=None, input_shape=(img_height, img_weight, channels))
    model_2 = setup_architechture_vgg16_2(model_2)
    model_train_2, history_2 = train_model_vgg16(checkpoint_path=check_point_path, save_weights_path=save_weight_two_path, model=model_2,
                                                 train_data=train_data, validation_data=validation_data, step_size_train=STEP_SIZE_TRAIN, step_size_valid=STEP_SIZE_VALID, epochs_train=Epochs)
    plot_result_train_model(history_2)

    # # test model
    # test_image = load_image(
    #     test_one_path, (img_height, img_weight), color_mode='rgb')
    # result_model_1 = model_train_1.predict(test_image)
    # result_model_2 = model_train_2.predict(test_image)

    # res_model_1 = np.argmax(result_model_1)
    # res_model_2 = np.argmax(result_model_2)
    # print("The predicted output model1 is :", dict_label[res_model_1])
    # print("The predicted output model2 is :", dict_label[res_model_2])
