import cv2
import pandas as pd
import numpy as np
import os

dict_exppression = {0: "anger", 1: "disgust", 2: "fear",
                    3: "happy", 4: "sadness", 5: "surprise", 6: "neutral"}

fer_data=pd.read_csv('./dataset/expression/fer2013.csv', delimiter=',')

if __name__ == "__main__":
    for index,row in fer_data.iterrows():
        pixels = np.asarray(list(row['pixels'].split(' ')),dtype=np.uint8)
        img = pixels.reshape((48,48))
        img = cv2.resize(img.astype('uint8'), (480, 480))
        label = dict_exppression[row['emotion']]
        pathname = os.path.join(f'./dataset/expression/fer_images/{str(label)}/{str(index)}.jpg')
        cv2.imwrite(pathname,img)
        print('image saved ias {}'.format(pathname))