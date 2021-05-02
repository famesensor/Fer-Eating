import cv2
import numpy as np
import multiprocessing as mp
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import multiprocessing
import os

from detection.object_detector import person_detect, face_detect, init_model_person_detect, init_model_face_detect
from preparation.preparation import load_image, load_vdo, resize_image, normalize_image
from behavior.behavior_model import init_model_behavior
from expression.expression_model import init_model_expression


# def expression_process(image: list):
#     # person detection
#     person_res = person_detect(net=person_model, image=frame)

#     # preparation data for behavior model
#     person_res = resize_image(
#         image=person_res, size_image=(img_height, img_width))
#     person_res = normalize_image(image=person_res)
#     person_res = np.expand_dims(person_res, axis=0)

#     # behavior detection
#     print("[INFO]: computing behavior detection...")
#     b_res = behavior_model.predict(person_res)
#     b_res = np.argmax(b_res)
#     behavior_res.append([nth_frame, dict_behavior[b_res]])


# def behavior_process(image: list):
#     # face dectection
#     face_res = face_detect(net=face_model, image=frame)

#     # preparation data for expression model
#     face_res = resize_image(
#         image=face_res, size_image=(img_height, img_width))
#     face_res = normalize_image(image=face_res)
#     face_res = np.expand_dims(face_res, axis=0)

#     # expression recognition
#     print("[INFO]: computing expression detection...")
#     e_res = expression_model.predict(face_res)
#     e_res = np.argmax(e_res)
#     expression_res.append([nth_frame, dict_exppression[e_res]])


if __name__ == "__main__":
    # init values...
    config_person = './models/yolo/yolov4.cfg'
    weight_person = './models/yolo/yolov4.weights'
    config_face = './models/dnn/deploy.prototxt.txt'
    weight_face = './models/dnn/res10_300x300_ssd_iter_140000_fp16.caffemodel'
    weight_behavior = "./models/behavior/vgg16_behavior_2.h5"
    weight_expression = "./models/expression/vgg16/vgg16_expression.h5"
    include_top = True
    class_num = 8
    number_frame = 0
    img_height, img_weight = 224, 224
    channels = 3  # 3 RGB
    batch_size = 32
    activation = "softmax"
    loss = "categorical_crossentropy"
    every_n_frame = 5
    flag_eat = False
    number_eat = 0
    count_step = 0
    dict_exppression = {0: "anger", 1: "contempt", 2: "disgust",
                        3: "fear", 4: "happay", 5: "neutral", 6: "sadness", 7: "surprise"}
    dict_behavior = {0: "eat", 1: "noeat"}
    behavior_res = []
    expression_res = []
    frame_start_eat = 0
    frame_end_eat = 0
    img_height, img_width = 224, 224

    # init model...
    person_model = init_model_person_detect(
        config=config_person, weight=weight_person)
    face_model = init_model_face_detect(config=config_face, weight=weight_face)
    behavior_model = init_model_behavior(weight_path=weight_behavior, types="vgg16", include_top=include_top, img_height=img_height,
                                         img_weight=img_weight, channels=channels, class_num=2, layer_num=19, activation=activation, loss=loss)
    expression_model = init_model_expression(weight_path=weight_expression, types="vgg16", include_top=include_top, img_height=img_height,
                                             img_weight=img_weight, channels=channels, class_num=class_num, layer_num=19, activation=activation, loss=loss)

    # load dataset...
    vdo_path = "./dataset/test/Deep1.MOV"
    vdocap = load_vdo(vdo_path=vdo_path)
    index = 0
    print("[INFO]: ID of main process: {}".format(os.getpid()))

    # sequence pattern
    while True:
        nth_frame = vdocap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = vdocap.read()

        if not ret:
            break

        print("[INFO]: skip frame : {}".format(every_n_frame))
        if nth_frame % every_n_frame == 0:
            print("[INFO]: frame no. {}".format(nth_frame))
            # preparation data
            # TODO: resize frame and normalization

            # person detection
            person_res = person_detect(net=person_model, image=frame)

            # preparation data for behavior model
            person_res = resize_image(
                image=person_res, size_image=(img_height, img_width))
            person_res = normalize_image(image=person_res)
            person_res = np.expand_dims(person_res, axis=0)

            # behavior detection
            print("[INFO]: computing behavior detection...")
            b_res = behavior_model.predict(person_res)
            b_res = np.argmax(b_res)

            # condition for change step frame rate
            if b_res == 0:
                if not flag_eat and number_eat == 0:
                    print(
                        "[INFO]: first eating in video frame on. {}".format(nth_frame))
                    flag_eat = True
                    number_eat = 1
                    every_n_frame = 1
                    frame_start_eat = nth_frame

                    file_name = './export/export_first_frame_eat_' + \
                        str(nth_frame)+'.jpg'
                    cv2.imwrite(file_name, frame)

            if flag_eat and number_eat == 1:
                print("[INFO]: after first eating...")
                count_step += 1
            if count_step == 100:
                print("[INFO]: change skip frame to default...")
                frame_end_eat = nth_frame
                every_n_frame = 5

            behavior_res.append([nth_frame, dict_behavior[b_res]])

            # face dectection
            face_res = face_detect(net=face_model, image=frame)

            # preparation data for expression model
            face_res = resize_image(
                image=face_res, size_image=(img_height, img_width))
            face_res = normalize_image(image=face_res)
            face_res = np.expand_dims(face_res, axis=0)

            # expression recognition
            print("[INFO]: computing expression detection...")
            e_res = expression_model.predict(face_res)
            e_res = np.argmax(e_res)
            expression_res.append([nth_frame, dict_exppression[e_res]])

    # TODO: plot result

    # # multiprocess
    # while True:
    #     nth_frame = vdocap.get(cv2.CAP_PROP_POS_FRAMES)
    #     ret, frame = vdocap.read()

    #     if not ret:
    #         break

    #     print("[INFO]: skip frame : {}".format(every_n_frame))
    #     if nth_frame % every_n_frame == 0:
    #         print("[INFO]: frame no. {}".format(nth_frame))

    #         # creating processes
    #         p1 = multiprocessing.Process(target=behavior_process, args=(frame))
    #         p2 = multiprocessing.Process(
    #             target=expression_process, args=(frame))

    #         # starting processes
    #         p1.start()
    #         p2.start()

    #         # process IDs
    #         print("ID of process p1: {}".format(p1.pid))
    #         print("ID of process p2: {}".format(p2.pid))

    #         # wait until processes are finished
    #         p1.join()
    #         p2.join()

    #         # both processes finished
    #         print("Both processes finished execution!")

    #         # check if processes are alive
    #         print("Process p1 is alive: {}".format(p1.is_alive()))
    #         print("Process p2 is alive: {}".format(p2.is_alive()))
    #         print(behavior_res)

    #         # # condition for change step frame rate
    #         # if behavior_res[index] == 0:
    #         #     if not flag_eat and number_eat == 0:
    #         #         print(
    #         #             "[INFO]: first eating in video frame on. {}".format(nth_frame))
    #         #         flag_eat = True
    #         #         number_eat = 1
    #         #         every_n_frame = 1
    #         #         frame_start_eat = nth_frame
    #         # if flag_eat and number_eat == 1:
    #         #     print("[INFO]: after first eating...")
    #         #     count_step += 1
    #         # if count_step == 100:
    #         #     print("[INFO]: change skip frame to default...")
    #         #     frame_end_eat = nth_frame
    #         #     every_n_frame = 5

    #         # index += 1
