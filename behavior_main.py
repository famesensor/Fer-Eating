import numpy as np

from preparation.preparation import load_data_set, load_image
from behavior.behavior_model import init_model_train_behavior, train_model, plot_result_train_model

if __name__ == "__main__":
    # init value
    img_height, img_weight = 224, 224
    channels = 3
    train_data_path = "./dataset/behavior/train"
    validation_data_path = "./dataset/behavior/validate"
    test_data_path = "./dataset/behavior/test"
    # test_one_path = "./data-set/behavior/test/test.jpg"
    check_point_path = "./models/behavior/working/"
    save_weight_vgg16_path = "./models/behavior/vgg16/vgg16_behavior.h5"
    save_weight_vgg19_path = "./models/behavior/vgg19/vgg19_behavior.h5"
    save_weight_resnet_path = "./models/behavior/resnet/resnet_behavior.h5"
    save_weight_mobilenet_path = "./models/behavior/mobilenet/mobilenet_behavior.h5"
    batch_size = 32
    Epochs = 18
    include_top = True
    class_num = 2
    activation = "softmax"
    loss = "categorical_crossentropy"
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
    # test_data = load_data_set(
    #     test_datagen_args, test_data_path, (img_height, img_weight), batch_size)

    STEP_SIZE_TRAIN = train_data.n//train_data.batch_size
    STEP_SIZE_VALID = validation_data.n//validation_data.batch_size
    # STEP_SIZE_TEST = test_data.n // test_data.batch_size

    vgg16 = init_model_train_behavior(types="vgg16", include_top=include_top, img_height=img_height,
                                      img_weight=img_weight, channels=channels, class_num=class_num, layer_num=19, activation=activation, loss=loss)
    vgg16_res, history_vgg16 = train_model(checkpoint_path=check_point_path+"vgg16_best.h5", save_weights_path=save_weight_vgg16_path, model=vgg16, train_data=train_data,
                                           validation_data=validation_data, step_size_train=STEP_SIZE_TRAIN, step_size_valid=STEP_SIZE_VALID, epochs_train=Epochs)
    plot_result_train_model(history_vgg16, "vgg16")

    vgg19 = init_model_train_behavior(types="vgg19", include_top=include_top, img_height=img_height,
                                      img_weight=img_weight, channels=channels, class_num=class_num, layer_num=22, activation=activation, loss=loss)
    vgg19_res, history_vgg19 = train_model(checkpoint_path=check_point_path+"vgg19_best.h5", save_weights_path=save_weight_vgg19_path, model=vgg19, train_data=train_data,
                                           validation_data=validation_data, step_size_train=STEP_SIZE_TRAIN, step_size_valid=STEP_SIZE_VALID, epochs_train=Epochs)
    plot_result_train_model(history_vgg19, "vgg19")

    resnet = init_model_train_behavior(types="resnet", include_top=include_top, img_height=img_height,
                                       img_weight=img_weight, channels=channels, class_num=class_num, layer_num=190, activation=activation, loss=loss)
    resnet_res, history_resnet = train_model(checkpoint_path=check_point_path+"resnet_best.h5", save_weights_path=save_weight_resnet_path, model=resnet, train_data=train_data,
                                             validation_data=validation_data, step_size_train=STEP_SIZE_TRAIN, step_size_valid=STEP_SIZE_VALID, epochs_train=Epochs)
    plot_result_train_model(history_resnet, "resnet")

    mobile_net = init_model_train_behavior(types="mobilenet", include_top=include_top, img_height=img_height,
                                           img_weight=img_weight, channels=channels, class_num=class_num, layer_num=154, activation=activation, loss=loss)
    mobile_net_res, history_mobile_net = train_model(checkpoint_path=check_point_path+"mobilenet_best.h5", save_weights_path=save_weight_mobilenet_path, model=mobile_net, train_data=train_data,
                                                     validation_data=validation_data, step_size_train=STEP_SIZE_TRAIN, step_size_valid=STEP_SIZE_VALID, epochs_train=Epochs)
    plot_result_train_model(history_mobile_net, "mobilenet")
