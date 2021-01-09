# USAGE
# python object-detector-yolo.py --image test.jpg

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import time

c_threshold = 0.5 # set threshold for bounding box values
nms = 0.4 # set threshold for non maximum supression
width = 416  # width of input image
height = 416 # height of input image

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

classesFile = "models/yolo/coco.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# initialize a list of colors to represent each possible class label
COLORS = np.random.randint(0, 255, size=(len(classes), 3),
	dtype="uint8")

# PATH to weight and config files
config = 'models/yolo/yolov4.cfg'
weight = 'models/yolo/yolov4.weights'

# Read the model using dnn
net = cv.dnn.readNetFromDarknet(config, weight)
image = cv.imread(args['image'])

(H,W) = image.shape[:2]

# Get the names of output layers
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# generate blob for image input to the network
blob = cv.dnn.blobFromImage(image,1/255,(416,416),swapRB=True, crop=False)
net.setInput(blob)

start = time.time()

layersOutputs = net.forward(ln)
print(layersOutputs)

end = time.time()

# print the time required
print( end- start)

boxes = []
confidences = []
classIDs = []

for output in layersOutputs:
	# loop over each of the detections
	for detection in output:
		# extract the class ID and confidence (i.e., probability) of
		# the current object detection
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]
 
		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > c_threshold:
			
			box = detection[0:4]* np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")
 
			# use the center (x, y)-coordinates to derive the top and
			# and left corner of the bounding box
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))
 
			# update our list of bounding box coordinates, confidences,
			# and class IDs
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)
			
# Remove unnecessary boxes using non maximum suppression
idxs = cv.dnn.NMSBoxes(boxes, confidences, c_threshold, nms)

if len(idxs) > 0:
	# loop over the indexes we are keeping
	for i in idxs.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])
 
		# draw a bounding box rectangle and label on the image
		color = [int(c) for c in COLORS[classIDs[i]]]
		cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
		cv.putText(image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)
 
# show the output image
cv.imshow("Image", image)
cv.waitKey(0)