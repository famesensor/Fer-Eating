from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
import numpy as np
import cv2
from PIL import Image
from tensorflow.python.saved_model import tag_constants
import colorsys
import random
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def init_model_person(weight_path: str):
    saved_model_loaded = tf.saved_model.load(
        weight_path, tags=[tag_constants.SERVING])
    return saved_model_loaded


def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def crop_image(image, bboxes, classes=read_class_names('./models/yolov4-tensorflow/assets/coco.names'), show_label=True):
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    out_boxes, out_scores, out_classes, num_boxes = bboxes
    for i in range(num_boxes[0]):
        if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes:
            continue
        coor = out_boxes[0][i]
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)
        coor = coor.astype("int")

        score = out_scores[0][i]
        class_ind = int(out_classes[0][i])
        if classes[class_ind] == "person":
            # print(class_ind, score, classes[class_ind])
            # fontScale = 0.5
            # bbox_color = colors[class_ind]
            # bbox_thick = int(0.6 * (image_h + image_w) / 600)
            # c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
            # cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
            image = image[coor[0]: coor[2], coor[1]:coor[3]]
            break

            # if show_label:
            #     bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            #     t_size = cv2.getTextSize(
            #         bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            #     c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            #     cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(
            #         c3[1])), bbox_color, -1)  # filled

            #     cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
            #                 fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    return image


def person_detect(image: list, saved_model_loaded) -> tuple:
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_data = cv2.resize(original_image, (416, 416))
    image_data = image_data / 255.

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    infer = saved_model_loaded.signatures['serving_default']

    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.25
    )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(),
                 valid_detections.numpy()]
    image = crop_image(original_image, pred_bbox)
    return image
