import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications import ResNet50V2, MobileNetV2
from preparation.preparation import load_image
from expression.expression_model import setup_network
from detection.object_detector import face_detect
from preparation.preparation import load_vdo, resize_image, normalize_image
from datetime import datetime

if __name__ == "__main__":
    # init value
    img_height, img_weight = 224, 224
    channels = 3  # 3 RGB
    batch_size = 32

    image_test_Path = "./data-set/behavior/test/506.jpg"
    weight_model_vgg16 = "./models/expression/vgg16/vgg16_expression.h5"
    weight_model_vgg19 = "./models/expression/vgg19/vgg19_expression.h5"
    weight_model_resnet = "./models/expression/resnet/resnet_expression.h5"
    weight_model_mobilenet = "./models/expression/mobilenet/mobilenet_expression.h5"
    vdo_path = "./dataset/test/Deep1.MOV"
    include_top = True
    class_num = 8
    activation = "softmax"
    loss = "categorical_crossentropy"
    dict_exppression = {0: "anger", 1: "contempt", 2: "disgust",
                        3: "fear", 4: "happay", 5: "neutral", 6: "sadness", 7: "surprise"}

    print("[INFO]: init vgg16 model...")
    # load model vgg16
    vgg16 = VGG16(include_top=include_top, input_tensor=None,
                  input_shape=(img_height, img_weight, channels))
    model_vgg16 = setup_network(model=vgg16, include_top=include_top,
                                class_num=class_num, layer_num=19, activation=activation, loss=loss)
    model_vgg16.load_weights(weight_model_vgg16)

    print("[INFO]: init vgg19 model...")
    # load model vgg19
    vgg19 = VGG19(include_top=include_top, input_tensor=None,
                  input_shape=(img_height, img_weight, channels))
    model_vgg19 = setup_network(model=vgg19, include_top=include_top,
                                class_num=class_num, layer_num=22, activation=activation, loss=loss)
    model_vgg19.load_weights(weight_model_vgg19)

    print("[INFO]: init resnet model...")
    # load model resnet
    resnet = ResNet50V2(include_top=include_top, input_tensor=None,
                        input_shape=(img_height, img_weight, channels))
    model_resnet = setup_network(model=resnet, include_top=include_top,
                                 class_num=class_num, layer_num=190, activation=activation, loss=loss)
    model_resnet.load_weights(weight_model_resnet)

    print("[INFO]: init moileNet model...")
    # load model mobilenet
    mobilenet = MobileNetV2(include_top=include_top, input_tensor=None, input_shape=(
        img_height, img_weight, channels))
    model_mobile = setup_network(model=mobilenet, include_top=include_top,
                                 class_num=class_num, layer_num=154, activation=activation, loss=loss)
    model_mobile.load_weights(weight_model_mobilenet)

    print("[INFO]: load vdo...")
    vdocap = load_vdo(vdo_path=vdo_path)

    print("[INFO]: Compare model expression...")
    test_face = []
    while True:
        nth_frame = vdocap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = vdocap.read()

        if not ret:
            break

        # face dectection
        face_res = face_detect(image=frame)

        # preparation data for expression model
        face_res = resize_image(
            image=face_res, size_image=(224, 224))
        face_res = normalize_image(image=face_res)
        face_res = np.expand_dims(face_res, axis=0)
        test_face.append(face_res)

    start = datetime.now()
    print("[INFO]: start time predict vgg16: ", start)
    res_vgg16 = model_vgg16.predict(test_face)
    duration = datetime.now() - start
    print("[INFO]: time predict vgg16 : ", duration)

    start = datetime.now()
    print("[INFO]: start time predict vgg19: ", start)
    res_vgg19 = model_vgg19.predict(test_face)
    duration = datetime.now() - start
    print("[INFO]: time predict vgg19 : ", duration)

    start = datetime.now()
    print("[INFO]: start time predict resnet: ", start)
    res_resnet = model_resnet.predict(test_face)
    duration = datetime.now() - start
    print("[INFO]: time predict resnet : ", duration)

    start = datetime.now()
    print("[INFO]: start time predict mobile: ", start)
    res_mobile = model_mobile.predict(test_face)
    duration = datetime.now() - start
    print("[INFO]: time predict mobile : ", duration)
    
    
