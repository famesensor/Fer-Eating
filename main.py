import cv2
import numpy as np
import multiprocessing as mp
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt

from detection.object_detector import person_detect, face_detect
from preparation.preparation import load_image, load_vdo, resize_image, normalize_image
from behavior.behavior_model import init_model_behavior
from expression.expression_model import init_model_expression

if __name__ == "__main__":
    # init value
    weight_behavior = "./models/behavior/vgg16_behavior_1.h5"
    weight_expression = "./models/expression/vgg16/vgg16_expression.h5"
    behavior_model = init_model_behavior(weight_path=weight_behavior)
    expression_model = init_model_expression(weight_path=weight_expression)
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

    # sequence pattern
    vdo_path = "./dataset/test/Deep1.MOV"
    vdocap = load_vdo(vdo_path=vdo_path)

    while True:
        nth_frame = vdocap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = vdocap.read()

        if not ret:
            break

        print("[INFO]: Skip frame : {}".format(every_n_frame))
        if nth_frame % every_n_frame == 0:
            print("[INFO]: frame no. {}".format(nth_frame))
            # preparation data
            # TODO: resize frame and normalization

            # person detection
            person_res = person_detect(image=frame)

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
            if flag_eat and number_eat == 1:
                print("[INFO]: after first eating...")
                count_step += 1
            if count_step == 100:
                print("[INFO]: change skip frame to default...")
                frame_end_eat = nth_frame
                every_n_frame = 5

            behavior_res.append([nth_frame, dict_behavior[b_res]])

            # face dectection
            face_res = face_detect(image=frame)

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

        # multiprocess
