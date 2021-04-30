import keras
import numpy as np
import cv2
import time
import glob
import matplotlib.pyplot as plt
from datetime import datetime
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications import ResNet50V2, MobileNetV2

from preparation.preparation import load_image
from expression.expression_model import init_model_expression
from detection.object_detector import face_detect, init_model_face_detect
from preparation.preparation import load_vdo, resize_image, normalize_image, load_data_set, load_image
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

    image_test_Path = "./data-set/behavior/test/506.jpg"
    weight_model_vgg16 = "./models/expression/vgg16/vgg16_expression.h5"
    weight_model_vgg19 = "./models/expression/vgg19/vgg19_expression.h5"
    weight_model_resnet = "./models/expression/resnet/resnet_expression.h5"
    weight_model_mobilenet = "./models/expression/mobilenet/mobilenet_expression.h5"
    config_face = './models/dnn/deploy.prototxt.txt'
    weight_face = './models/dnn/res10_300x300_ssd_iter_140000_fp16.caffemodel'
    data_evalute_path = './dataset/validate/'
    data_time_path = './dataset/unlabel/*.png'
    vdo_path = "./dataset/test/Deep1.MOV"
    include_top = True
    class_num = 8
    number_frame = 0
    activation = "softmax"
    loss = "categorical_crossentropy"
    dict_exppression = {0: "anger", 1: "contempt", 2: "disgust",
                        3: "fear", 4: "happay", 5: "neutral", 6: "sadness", 7: "surprise"}
    test_datagen_args = dict(
        rescale=1./255
    )

    # # init model...
    face_model = init_model_face_detect(config=config_face, weight=weight_face)

    # vgg16 = init_model_expression(weight_path=weight_model_vgg16, types="vgg16", include_top=include_top, img_height=img_height,
    #                               img_weight=img_weight, channels=channels, class_num=class_num, layer_num=19, activation=activation, loss=loss)

    # vgg19 = init_model_expression(weight_path=weight_model_vgg19, types="vgg19", include_top=include_top, img_height=img_height,
    #                               img_weight=img_weight, channels=channels, class_num=class_num, layer_num=22, activation=activation, loss=loss)

    # resnet = init_model_expression(weight_path=weight_model_resnet, types="resnet", include_top=include_top, img_height=img_height,
    #                                img_weight=img_weight, channels=channels, class_num=class_num, layer_num=190, activation=activation, loss=loss)

    mobile_net = init_model_expression(weight_path=weight_model_mobilenet, types="mobilenet", include_top=include_top, img_height=img_height,
                                        img_weight=img_weight, channels=channels, class_num=class_num, layer_num=154, activation=activation, loss=loss)

    print("[INFO]: load vdo...")
    vdocap = load_vdo(vdo_path=vdo_path)

    print("[INFO]: compare time model expression...")
    start = datetime.now()
    startTime = time.time()
    print("[INFO]: start time predict vgg16: ", start)
    images = glob.glob(data_time_path)
    res_predict = []
    for image_path in images:
        image = cv2.imread(image_path)

        # face dectection
        face_res = face_detect(net=face_model, image=image)

        # preparation data for expression model
        face_res = resize_image(
            image=face_res, size_image=(img_height, img_weight))
        face_res = normalize_image(image=face_res)
        face_res = np.expand_dims(face_res, axis=0)

        # res_predict.append(vgg16.predict(face_res))
        # res_predict.append(vgg19.predict(face_res))
        # res_predict.append(resnet.predict(face_res))
        res_predict.append(mobile_net.predict(face_res))

    # Time elapsed
    end = time.time()
    duration = datetime.now() - start
    print("[INFO]: time predict Mobile Net : ", duration)
    seconds = end - startTime
    fps = len(images)/seconds
    print("[INFO]: Estimated frames per second : {0}".format(fps))

    print("[INFO]: compare evaluate model expression...")
    test_data = load_data_set(
        test_datagen_args, data_evalute_path, (img_height, img_weight), batch_size)

    # vgg16_evaluate[0] is loss, vgg16_evaluate[1] is accuracy
    # vgg16_evaluate = vgg16.evaluate(test_data)

    # vgg19_evaluate = vgg19.evaluate(test_data)

    # resnet_evaluate = resnet.evaluate(test_data)

    mobile_evaluate = mobile_net.evaluate(test_data)
    
    # Evaluation on test dataset
    # test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size=16)
    print("test loss, test acc:", mobile_evaluate)
    # print("Loss on test set: ", test_loss)
    # print("Accuracy on test set: ", test_acc)
    
    # Plot the confusion matrix
    test_logits = mobile_net.predict(test_data)
    rounded_labels=np.argmax(test_logits, axis=1)
    # rounded_labels.sort()

    cm  = confusion_matrix(test_data.labels, np.round(rounded_labels))
    class_names=['anger', 'contempt', 'disgust', 'fear', 'happay', 'neutral', 'sadness', 'surprise']
    # Plot Non-Normalized
    plot_confusion_matrix(cm, class_names=class_names)
    # # Plot Normalized
    # plot_confusion_matrix(cm, show_absolute=False, show_normed=True, class_names=class_names)
    plt.show()
 
    print("Accuracy Score :", accuracy_score(test_data.labels, rounded_labels))
   
    print("Recall Score : ", recall_score(test_data.labels, rounded_labels, average='macro'))
 
    print("Precision Score : ", precision_score(test_data.labels, rounded_labels, average='macro'))
 
    print("F1 Score : ", f1_score(test_data.labels, rounded_labels, average='macro'))
    
    # print(*rounded_labels)
    # print(len(rounded_labels))
     
    # print(*test_data.labels)
    # print(len(test_data.labels))
    
    # (rounded_labels==test_data.labels).all()
    
    # np.array_equal(rounded_labels,test_data.labels) 
    
    # import collections, numpy
    # collections.Counter(rounded_labels)
    # collections.Counter(test_data.labels)
    
    # (rounded_labels==test_data.labels).all()
