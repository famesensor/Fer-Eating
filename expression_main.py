from expression_model import init_model_vgg16, init_model_vgg19, init_model_resnet50v2, init_model_alexnet, init_model_mobilenet, setup_network, train_model, plot_result_train_model

if __name__ == "__main__":
    # init value
    img_height, img_weight = 224, 224
    channels = 3
    train_data_path = "./data-set/expression/train/"
    validation_data_path = "./data-set/expression/validate/"
    # test_data_path = "./data-set/behavior/test"
    # test_one_path = "./data-set/behavior/test/test.jpg"
    check_point_path = "./models/expression/working/"
    save_weight_vgg16 = "./models/expression/vgg16/vgg16_expression.h5"
    save_weight_vgg19 = "./models/expression/vgg19/vgg19_expression.h5"
    save_weight_resnet = "./models/expression/resnet/resnet_expression.h5"
    save_weight_alexnet = "./models/expression/alexnet/alexnet_expression.h5"
    save_weight_mobilenet = "./models/expression/mobilenet/mobilenet_expression.h5"
    batch_size = 32
    Epochs = 18
    include_top = True
    class_num = 8
    activation = "softmax"
    loss = "categorical_crossentropy"

    # load data set train,test and validate

    # vgg16
    print("vgg16 model start...")
    # init model
    vgg16_model = init_model_vgg16(include_top=include_top, input_tensor=None, input_shape=(
        img_height, img_weight, channels))
    # setup network
    vgg16_model = setup_network(
        model=vgg16_model, include_top=include_top, class_num=class_num, layer_num=19, activation=activation, loss=loss)
    # train model
    train_vgg16, history_vgg16 = train_model(checkpoint_path=check_point_path, save_weight_path=save_weight_vgg16,
                                             model=vgg16_model, train_data="", validation_data="", step_size_train="", step_size_valid="", epochs_train=Epochs)
    # plot result trian model
    plot_result_train_model(history=history_vgg16,
                            model_name="vgg16 accurency")
    print("vgg16 model end...")

    # vgg19
    print("vgg19 model start...")
    # init vgg19
    vgg19_model = init_model_vgg19(include_top=include_top, input_tensor=None, input_shape=(
        img_height, img_weight, channels))
    # setup network
    vgg19_model = setup_network(
        model=vgg19_model, include_top=include_top, class_num=class_num, layer_num=22, activation=activation, loss=loss)
    # train model
    train_vgg19, history_vgg19 = train_model(checkpoint_path=check_point_path, save_weight_path=save_weight_vgg19,
                                             model=vgg19_model, train_data="", validation_data="", step_size_train="", step_size_valid="", epochs_train=Epochs)
    # plot result trian model
    plot_result_train_model(history=history_vgg19,
                            model_name="vgg19 accurency")
    print("vgg19 model end...")

    # resnet
    print("resnet model start...")
    # init resnet
    resnet_model = init_model_resnet50v2(
        include_top=include_top, input_tensor=None, input_shape=(img_height, img_weight, channels))
    # setup network
    resnet_model = setup_network(
        model=resnet_model, include_top=include_top, class_num=class_num, layer_num=190, activation=activation, loss=loss)
    # train model
    train_resnet, history_resnet = train_model(checkpoint_path=check_point_path, save_weight_path=save_weight_resnet,
                                               model=resnet_model, train_data="", validation_data="", step_size_train="", step_size_valid="", epochs_train=Epochs)
    # plot result trian model
    plot_result_train_model(history=history_resnet,
                            model_name="resnet accurency")
    print("resnet model end...")

    # alexnet
    # init alexnet
    alexnet_model = init_model_alexnet()
    # mobilenet v2
    print("mobilenetV2 model start...")
    # init mobilenet
    mobile_model = init_model_mobilenet(
        include_top=include_top, input_tensor=None, input_shape=(img_height, img_weight, channels))
    # setup network
    mobile_model = setup_network(
        model=mobile_model, include_top=include_top, class_num=class_num, layer_num=154, activation=activation, loss=loss)
    # train model
    # train model
    train_resnet, history_resnet = train_model(checkpoint_path=check_point_path, save_weight_path=save_weight_mobilenet,
                                               model=mobile_model, train_data="", validation_data="", step_size_train="", step_size_valid="", epochs_train=Epochs)
    # plot result trian model
    plot_result_train_model(history=history_resnet,
                            model_name="mobilenet accurency")
    print("mobilenetV2 model end...")
