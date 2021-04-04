from object_detector import face_detect
import glob
import cv2

if __name__ == "__main__":
    images = glob.glob("dataset/expression/train/*/*.png")
    for image_path in images:
        print(image_path)
        # image = cv2.imread(image_path)
        # result = face_detect(image_path)
        # cv2.imwrite(image_path, result)

