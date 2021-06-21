import cv2
import math
import numpy as np
import multiprocessing as mp
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import multiprocessing
import os
import argparse
from datetime import datetime

from detection.object_detector import face_detect, init_model_face_detect
from detection.yolo_detector import init_model_person, person_detect
from preparation.preparation import load_image, load_vdo, resize_image, normalize_image
from behavior.behavior_model import init_model_behavior
from expression.expression_model import init_model_expression
from plot.plot import plot_graph


def export_image(image: list, frame_number: int) -> None:
    file_name = './export/export_first_frame_eat_' + \
        str(nth_frame)+'.jpg'
    image_list.append([nth_frame, file_name])
    cv2.imwrite(file_name, frame)
    return


def save_output(data: list, file_name: str) -> None:
    with open(f"./export/temp/{file_name}.txt", "w") as output:
        output.write(str(data))
    return


if __name__ == "__main__":

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True, help="path to the video file")
    args = vars(ap.parse_args())

    # init values...
    weight_person = './models/yolov4-tensorflow'
    config_face = './models/dnn/deploy.prototxt.txt'
    weight_face = './models/dnn/res10_300x300_ssd_iter_140000_fp16.caffemodel'
    weight_behavior = "./models/behavior/mobilenet/mobilenet_behavior.h5"
    weight_expression = "./models/expression/vgg16_5_best_d04.h5"
    include_top = False
    class_num_behavior = 2
    class_num_expression = 8
    img_height, img_width = 224, 224
    channels = 3  # 3 RGB
    batch_size = 32
    activation = "softmax"
    loss = "categorical_crossentropy"
    dict_exppression = {0: "anger", 1: "contempt", 2: "disgust",
                        3: "fear", 4: "happy", 5: "neutral", 6: "sadness", 7: "surprise"}
    dict_behavior = {0: "eat", 1: "noeat"}
    behavior_list = []
    expression_list = []
    image_list = []
    interest_area = []
    count_group = 0
    count_step = 0
    is_eating = False

    start = datetime.now()

    # init model...
    person_model = init_model_person(weight_path=weight_person)
    face_model = init_model_face_detect(config=config_face, weight=weight_face)
    behavior_model = init_model_behavior(weight_path=weight_behavior, types="mobilenet", include_top=include_top, img_height=img_height,
                                         img_weight=img_width, channels=channels, class_num=class_num_behavior, layer_num=19, activation=activation, loss=loss)
    expression_model = init_model_expression(weight_path=weight_expression, types="vgg16", include_top=include_top, img_height=img_height,
                                             img_weight=img_width, channels=channels, class_num=class_num_expression, layer_num=19, activation=activation, loss=loss)

    # load dataset...
    vdo_path = args['video']
    vdocap = load_vdo(vdo_path=vdo_path)
    fps = math.ceil(vdocap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        exit()

    interest_fps = fps * 5

    while True:
        nth_frame = vdocap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = vdocap.read()

        if not ret:
            break

        print("==============================================\n")
        print("[INFO]: frame no. {}".format(nth_frame))

        # person detection
        person_res = person_detect(
            image=frame, saved_model_loaded=person_model)

        if person_res is None:
            print("[INFO]: Person not found")
            print("[INFO]: continue...")
            continue

        if not person_res.any():
            print("[INFO]: Person not found")
            print("[INFO]: continue...")
            continue

        # face dectection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_img = np.zeros_like(frame)
        gray_img[:,:,0] = gray
        gray_img[:,:,1] = gray
        gray_img[:,:,2] = gray

        face_res = face_detect(net=face_model, image=gray_img)

        if face_res is None:
            print("[INFO]: Face not found")
            print("[INFO]: continue...")
            continue

        if not face_res.any():
            print("[INFO]: Face not found")
            print("[INFO]: continue...")
            continue

        # preparation data for behavior model
        person_res = resize_image(
            image=person_res, size_image=(img_height, img_width))
        person_res = normalize_image(image=person_res)
        person_res = np.expand_dims(person_res, axis=0)

        # behavior detection
        behavior_res = behavior_model.predict(person_res)
        behavior_res = np.argmax(behavior_res)

        ##
        ## condition for keep interest interval...
        ##
        # checking if behavior response is "eat"
        if behavior_res == 0:
            # checking for eating activity
            if not is_eating: 
                print(
                    "\n[INFO]: First eating in video frame on. {}".format(nth_frame))

                is_eating = True

                # export image of first eating for plot
                export_image(frame, nth_frame) 

        # checking if sample is eating
        if is_eating:
            # append group in this eating phase
            # and counting frame
            interest_area.append([nth_frame, count_group])
            count_step += 1
        else:
            # group -1 when sample not in eating phase
            interest_area.append([nth_frame, -1])

        # checking if counting was more than 2 seconds
        if count_step >= interest_fps:
            # reset variables and move to next group
            is_eating = False
            count_group += 1
            count_step = 0

        behavior_list.append([nth_frame, dict_behavior[behavior_res]])
        print(f"[BEHAVIOR]: {dict_behavior[behavior_res]}")

        # preparation data for expression model
        face_res = resize_image(
            image=face_res, size_image=(img_height, img_width))
        face_res = normalize_image(image=face_res)
        face_res = np.expand_dims(face_res, axis=0)

        # expression recognition
        expression_res = expression_model.predict(face_res)
        expression_res = np.argmax(expression_res)
        expression_list.append([nth_frame, dict_exppression[expression_res]])
        print(f"[EXPRESSION]: {dict_exppression[expression_res]}")
        print("\n==============================================")

    save_output(expression_list, 'expression')
    save_output(behavior_list, 'behavior')
    save_output(image_list, 'image')
    save_output(interest_area, 'interest_area')

    duration = datetime.now() - start
    print("Completed in time: ", duration)

    # plot result
    plot_graph(graph_type='level')
