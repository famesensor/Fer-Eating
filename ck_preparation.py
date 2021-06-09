from detection.object_detector import face_detect, init_model_face_detect
import glob
import cv2

if __name__ == "__main__":
    config_face = './models/dnn/deploy.prototxt.txt'
    weight_face = './models/dnn/res10_300x300_ssd_iter_140000_fp16.caffemodel'
    face_model = init_model_face_detect(config=config_face, weight=weight_face)


    images = glob.glob("dataset/expression/test/*/*.tiff")
    for image_path in images:
        print(image_path)
        image = cv2.imread(image_path)
        result = face_detect(net=face_model, image=image)
        cv2.imwrite(image_path, result)

