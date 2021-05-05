import cv2
import time
import glob
import numpy as np
from datetime import datetime

from behavior.behavior_model import init_model_behavior
from detection.object_detector import init_model_person_detect, person_detect
from preparation.preparation import resize_image, normalize_image, load_data_set

if __name__ == "__main__":
    # init value
    img_height, img_weight = 224, 224
    channels = 3  # 3 RGB
    batch_size = 32

    weight_model_vgg16 = "./models/behavior/vgg16/vgg16_behavior.h5"
    weight_model_vgg19 = "./models/behavior/vgg19/vgg19_behavior.h5"
    weight_model_resnet = "./models/behavior/resnet/resnet_behavior.h5"
    weight_model_mobilenet = "./models/behavior/mobilenet/mobilenet_behavior.h5"
    config_person = './models/yolo/yolov4.cfg'
    weight_person = './models/yolo/yolov4.weights'
    include_top = True
    class_num = 2
    activation = "softmax"
    loss = "categorical_crossentropy"
    data_time_path = './dataset/behavior/test_time/'
    data_evalute_path = './dataset/behavior/validate/'
    test_datagen_args = dict(
        rescale=1./255
    )

    # init model...
    person_model = init_model_person_detect(
        config=config_person, weight=weight_person)

    vgg16 = init_model_behavior(weight_path=weight_model_vgg16, types="vgg16", include_top=include_top, img_height=img_height,
                                img_weight=img_weight, channels=channels, class_num=class_num, layer_num=19, activation=activation, loss=loss)

    vgg19 = init_model_behavior(weight_path=weight_model_vgg16, types="vgg19", include_top=include_top, img_height=img_height,
                                img_weight=img_weight, channels=channels, class_num=class_num, layer_num=22, activation=activation, loss=loss)

    resnet = init_model_behavior(weight_path=weight_model_vgg16, types="vgg19", include_top=include_top, img_height=img_height,
                                 img_weight=img_weight, channels=channels, class_num=class_num, layer_num=190, activation=activation, loss=loss)

    mobile_net = init_model_behavior(weight_path=weight_model_vgg16, types="mobilenet", include_top=include_top, img_height=img_height,
                                     img_weight=img_weight, channels=channels, class_num=class_num, layer_num=154, activation=activation, loss=loss)

    print("[INFO]: compare time model expression...")
    start = datetime.now()
    startTime = time.time()
    print("[INFO]: start time predict vgg16: ", start)
    images = glob.glob(data_time_path)
    res_predict = []
    test_data = []
    for image_path in images:
        image = cv2.imread(image_path)
        test_data.append(image)
        # face dectection
        person_res = person_detect(net=person_detect, image=image)

        # preparation data for expression model
        preson_res = resize_image(
            image=preson_res, size_image=(img_height, img_weight))
        preson_res = normalize_image(image=preson_res)
        facepreson_res_res = np.expand_dims(preson_res, axis=0)

        res_predict.append(vgg16.predict(preson_res))
        # res_predict.append(vgg19.predict(face_res))
        # res_predict.append(resnet.predict(face_res))
        # res_predict.append(mobile_net.predict(face_res))

     # Time elapsed
    end = time.time()
    duration = datetime.now() - start
    print("[INFO]: time predict vgg16 : ", duration)
    seconds = end - startTime
    fps = len(images)/seconds
    print("[INFO]: Estimated frames per second : {0}".format(fps))

    print("[INFO]: compare evaluate model expression...")
    test_data = load_data_set(
        test_datagen_args, data_evalute_path, (img_height, img_weight), batch_size)

    # vgg16_evaluate[0] is loss, vgg16_evaluate[1] is accreency
    vgg16_evaluate = vgg16.evaluate(test_data)

    vgg19_evaluate = vgg19.evaluate(test_data)

    resnet_evaluate = resnet.evaluate(test_data)

    mobile_evaluate = mobile_net.evaluate(test_data)
