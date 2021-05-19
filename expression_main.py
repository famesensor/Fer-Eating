from expression.expression_model import init_model_resnet50v2, init_model_alexnet, init_model_mobilenet, setup_network, init_model_train_expression, train_model, plot_result_train_model
from preparation.preparation import load_data_set

if __name__ == "__main__":
    # init value
    img_height, img_width = 224, 224
    channels = 3
    train_data_path = "./dataset/expression/train/"
    validation_data_path = "./dataset/expression/validate/"
    check_point_path = "./models/expression/working/"
    save_weight_vgg16 = "./models/expression/vgg16/vgg16_expression.h5"
    save_weight_vgg19 = "./models/expression/vgg19/vgg19_expression.h5"
    save_weight_resnet = "./models/expression/resnet/resnet_expression.h5"
    save_weight_alexnet = "./models/expression/alexnet/alexnet_expression.h5"
    save_weight_mobilenet = "./models/expression/mobilenet/mobilenet_expression.h5"
    batch_size = 32
    Epochs = 18
    include_top = False
    class_num = 8
    dropout = 0.2
    activation = "softmax"
    loss = "categorical_crossentropy"
    train_datagen_args = dict(
        rotation_range=20,
        rescale=1./255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    # load data set train,test and validate
    print("load data...")
    train_data = load_data_set(
        train_datagen_args, train_data_path, (img_height, img_width), batch_size)
    validation_data = load_data_set(
        train_datagen_args, validation_data_path, (img_height, img_width), batch_size)

    STEP_SIZE_TRAIN = train_data.n//train_data.batch_size
    STEP_SIZE_VALID = validation_data.n//validation_data.batch_size
    print("load data end...")

    # vgg16
    print("vgg16 model start...")
    # init model
    vgg16_model = init_model_train_expression(types="vgg16", include_top=include_top, img_height=img_height, img_width=img_width,
                                              channels=channels, class_num=class_num, layer_num=19, activation=activation, loss=loss, dropout=dropout)
    # # train model
    # train_vgg16, history_vgg16 = train_model(checkpoint_path=check_point_path+"vgg16_best.h5", save_weight_path=save_weight_vgg16,
    #                                          model=vgg16_model, train_data=train_data, validation_data=validation_data, step_size_train=STEP_SIZE_TRAIN, step_size_valid=STEP_SIZE_VALID, epochs_train=Epochs)
    # # plot result trian model
    # plot_result_train_model(history=history_vgg16,
    #                         model_name="vgg16 accurency")
    # print("vgg16 model end...")

    # vgg19
    print("vgg19 model start...")
    # init vgg19
    vgg19_model = init_model_train_expression(types="vgg19", include_top=include_top, img_height=img_height, img_width=img_width,
                                              channels=channels, class_num=class_num, layer_num=22, activation=activation, loss=loss, dropout=dropout)
    # # train model
    # train_vgg19, history_vgg19 = train_model(checkpoint_path=check_point_path+"vgg19_best.h5", save_weight_path=save_weight_vgg19,
    #                                          model=vgg19_model, train_data=train_data, validation_data=validation_data, step_size_train=STEP_SIZE_TRAIN, step_size_valid=STEP_SIZE_VALID, epochs_train=Epochs)
    # # plot result trian model
    # plot_result_train_model(history=history_vgg19,
    #                         model_name="vgg19 accurency")
    # print("vgg19 model end...")

    # resnet
    print("resnet model start...")
    # init resnet
    resnet_model = init_model_train_expression(types="resnet", include_top=include_top, img_height=img_height, img_width=img_width,
                                               channels=channels, class_num=class_num, layer_num=190, activation=activation, loss=loss, dropout=dropout)
    # # train model
    # train_resnet, history_resnet = train_model(checkpoint_path=check_point_path, save_weight_path=save_weight_resnet,
    #                                            model=resnet_model, train_data=train_data, validation_data=validation_data, step_size_train=STEP_SIZE_TRAIN, step_size_valid=STEP_SIZE_VALID, epochs_train=Epochs)
    # # plot result trian model
    # plot_result_train_model(history=history_resnet,
    #                         model_name="resnet accurency")
    # print("resnet model end...")

    # # alexnet
    # # init alexnet
    # print("alexnet model start...")
    # alexnet_model = init_model_alexnet()
    # # train model
    # train_alexnet, history_alexnet = train_model(checkpoint_path=check_point_path, save_weight_path=save_weight_alexnet, model=alexnet_model,
    #                                              train_data=train_data, validation_data=validation_data, step_size_train=STEP_SIZE_TRAIN, step_size_valid=STEP_SIZE_VALID, epochs_train=Epochs)
    # plot_result_train_model(history=history_alexnet,
    #                         model_name="alexnet accurency")
    # print("alexnet model end...")

    # # mobilenet v2
    # print("mobilenetV2 model start...")
    # # init mobilenet
    # mobile_model = init_model_mobilenet(
    #     include_top=include_top, input_tensor=None, input_shape=(img_height, img_weight, channels))
    # # setup network
    # mobile_model = setup_network(
    #     model=mobile_model, include_top=include_top, class_num=class_num, layer_num=154, activation=activation, loss=loss)
    # # train model
    # train_resnet, history_resnet = train_model(checkpoint_path=check_point_path, save_weight_path=save_weight_mobilenet,
    #                                            model=mobile_model, train_data=train_data, validation_data=validation_data, step_size_train=STEP_SIZE_TRAIN, step_size_valid=STEP_SIZE_VALID, epochs_train=Epochs)
    # # plot result trian model
    # plot_result_train_model(history=history_resnet,
    #                         model_name="mobilenet accurency")
    # print("mobilenetV2 model end...")
