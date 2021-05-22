import cv2
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from behavior.behavior_model import init_model_behavior
from detection.yolo_detector import init_model_person, person_detect
from preparation.preparation import resize_image, normalize_image, load_data_set
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


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
    include_top = False
    class_num = 2
    activation = "softmax"
    loss = "categorical_crossentropy"
    data_time_path = './dataset/behavior/test_time/*.jpg'
    data_evalute_path = './dataset/behavior/validate/'
    test_datagen_args = dict(
        rescale=1./255
    )

    # init model...
    person_model = init_model_person("./models/yolov4-tensorflow")
    vgg16 = init_model_behavior(weight_path=weight_model_vgg16, types="vgg16", include_top=include_top, img_height=img_height,
                                img_weight=img_weight, channels=channels, class_num=class_num, layer_num=19, activation=activation, loss=loss)

    # vgg19 = init_model_behavior(weight_path=weight_model_vgg19, types="vgg19", include_top=include_top, img_height=img_height,
    #                             img_weight=img_weight, channels=channels, class_num=class_num, layer_num=22, activation=activation, loss=loss)

    # resnet = init_model_behavior(weight_path=weight_model_resnet, types="resnet", include_top=include_top, img_height=img_height,
    #                               img_weight=img_weight, channels=channels, class_num=class_num, layer_num=190, activation=activation, loss=loss)

    # mobile_net = init_model_behavior(weight_path=weight_model_mobilenet, types="mobilenet", include_top=include_top, img_height=img_height,
    #                                   img_weight=img_weight, channels=channels, class_num=class_num, layer_num=154, activation=activation, loss=loss)

    print("[INFO]: compare time model expression...")
    start = datetime.now()
    startTime = time.time()
    print("[INFO]: start time predict vgg19: ", start)
    images = glob.glob(data_time_path)
    res_predict = []
    test_data = []
    index = 0
    for image_path in images:
        image = cv2.imread(image_path)

        # face dectection
        person_res = person_detect(image, person_model)
        if not person_res.any():
            continue

        # preparation data for expression model
        person_res = resize_image(
            image=person_res, size_image=(img_height, img_weight))
        person_res = normalize_image(image=person_res)
        person_res = np.expand_dims(person_res, axis=0)

        res_predict.append(vgg16.predict(person_res))
        # res_predict.append(vgg19.predict(person_res))
        # res_predict.append(resnet.predict(person_res))
        # res_predict.append(mobile_net.predict(person_res))
        print(image_path)

    # Time elapsed
    end = time.time()
    duration = datetime.now() - start
    print("[INFO]: time predict vgg19 : ", duration)
    seconds = end - startTime
    fps = len(res_predict)/seconds
    print(f"[INFO]: images : {len(res_predict)}")
    print("[INFO]: Estimated frames per second : {0}".format(fps))

    # # print("[INFO]: compare evaluate model expression...")
    # test_data = load_data_set(
    #     test_datagen_args, data_evalute_path, (img_height, img_weight), batch_size)

    # # vgg16_evaluate[0] is loss, vgg16_evaluate[1] is accreency
    # # vgg16_evaluate = vgg16.evaluate(test_data)

    # vgg19_evaluate = vgg19.evaluate(test_data)

    # # resnet_evaluate = resnet.evaluate(test_data)

    # # mobile_evaluate = mobile_net.evaluate(test_data)

    # # Evaluation on test dataset
    # print("test loss, test acc:", vgg19_evaluate)

    # # Plot the confusion matrix
    # test_logits = vgg19.predict(test_data)
    # rounded_labels = np.argmax(test_logits, axis=1)

    # cm = confusion_matrix(test_data.classes, np.round(rounded_labels))
    # class_names = ['eat', 'dont eat']
    # # Plot Non-Normalized
    # plot_confusion_matrix(cm, class_names=class_names)
    # # Plot Normalized
    # plot_confusion_matrix(cm, show_absolute=False,
    #                       show_normed=True, class_names=class_names)
    # plt.show()

    # print("Accuracy Score :", accuracy_score(test_data.labels, rounded_labels))

    # print("Recall Score : ", recall_score(
    #     test_data.labels, rounded_labels, average='macro'))

    # print("Precision Score : ", precision_score(
    #     test_data.labels, rounded_labels, average='macro'))

    # print("F1 Score : ", f1_score(
    #     test_data.labels, rounded_labels, average='macro'))
