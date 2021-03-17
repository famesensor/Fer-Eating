from expression_model import init_model_vgg16, init_model_vgg19, init_model_resnet50v2, init_model_mobilenet, setup_network_vgg16, setup_network_vgg19, setup_network_resnet, setup_network_mobilenet

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

    # load data set train,test and validate

    # init vgg16
    vgg16_model = init_model_vgg16(include_top=include_top, input_tensor=None, input_shape=(
        img_height, img_weight, channels))
    vgg16_model = setup_network_vgg16(
        model=vgg16_model, include_top=include_top, class_num=class_num)

    # init vgg19
    vgg19_model = init_model_vgg19(include_top=include_top, input_tensor=None, input_shape=(
        img_height, img_weight, channels))
    vgg19_model = setup_network_vgg19(
        model=vgg19_model, include_top=include_top, class_num=chr)

    # init resnet
    resnet_model = init_model_resnet50v2(
        include_top=include_top, input_tensor=None, input_shape=(img_height, img_weight, channels))
    resnet_model = setup_network_resnet(
        model=resnet_model, include_top=include_top, class_num=class_num)

    # init alexnet

    # init mobilenet
    mobile_model = init_model_mobilenet(
        include_top=include_top, input_tensor=None, input_shape=(img_height, img_weight, channels))
    mobile_model = setup_network_mobilenet(
        model=mobile_model, include_top=include_top, class_num=class_num)
