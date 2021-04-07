# USAGE
# python detector.py --detect face --image person.jpg

import cv2
import argparse
import sys
import numpy as np
import os.path
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-d", "--detect", required=True,
                help="detect 'person' or 'face'")
ap.add_argument("-t", "--threshold", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
threshold = args["threshold"]

def draw(boxes, confidences):
    for i, box in enumerate(boxes):

        (startX, startY) = (box[0], box[1])
        (endX, endY) = (box[2], box[3])

        # draw a bounding box rectangle and label on the image
        color = (0, 0, 255)
        text = "{:.4f}".format(confidences[i])
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 2)

    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)

def crop(boxes):

    for i, box in enumerate(boxes):

        (startX, startY) = (box[0], box[1])
        (endX, endY) = (box[2], box[3])

        crop_img = image[startY:endY, startX:endX]

        # show the output image
        cv2.imshow("Image", crop_img)
        cv2.waitKey(0)

def personDetect(image, threshold):

    nms = 0.4  # set threshold for non maximum supression
    width = 416  # width of input image
    height = 416  # height of input image

    # PATH to weight and config files
    config = 'models/yolo/yolov4.cfg'
    weight = 'models/yolo/yolov4.weights'

    classesFile = "models/yolo/coco.names"
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    # Read the model
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromDarknet(config, weight)

    # Get the names of output layers
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # generate blob for image input to the network
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(
        image, (width, height)), 1/255, (width, height), swapRB=True, crop=False)
    net.setInput(blob)

    print("[INFO] computing person detections...")
    start = time.time()

    layersOutputs = net.forward(ln)

    end = time.time()

    # print the time required
    print("usage:", end - start, "sec")

    boxes = []
    confidences = []
    classIDs = []
    classPerson = 0

    for output in layersOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classPerson]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > threshold:

                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                startX = int(centerX - (width / 2))
                startY = int(centerY - (height / 2))

                endX = int(startX + width)
                endY = int(startY + height)

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([startX, startY, endX, endY])
                confidences.append(float(confidence))

    # Remove unnecessary boxes using non maximum suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, nms)

    boxes_new = []
    confidences_new = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            boxes_new.append([boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]])
            confidences_new.append(float(confidences[i]))

    # draw bounding boxes
    # draw(boxes_new, confidences_new)

    crop(boxes_new)


def faceDetect(image, threshold):

    width = 300  # width of input image
    height = 300  # height of input image

    # PATH to weight and config files
    config = 'models/dnn/deploy.prototxt.txt'
    weight = 'models/dnn/res10_300x300_ssd_iter_140000_fp16.caffemodel'

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(config, weight)

    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (width, height)), 1.0,
                                 (width, height), (104.0, 177.0, 123.0))
    net.setInput(blob)

    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing face detections...")
    start = time.time()

    detections = net.forward()

    end = time.time()

    # print the time required
    print("usage:", end - start, "sec")

    detections = np.squeeze(detections)

    boxes = []
    confidences = []

    # loop over the detections
    for i in range(0, detections.shape[0]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > threshold:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            boxes.append([startX, startY, endX, endY])
            confidences.append(float(confidence))

    # draw bounding boxes
    draw(boxes, confidences)


if(args["detect"] == 'person'):
    personDetect(image, threshold)
elif (args["detect"] == 'face'):
    faceDetect(image, threshold)
else:
    pass
