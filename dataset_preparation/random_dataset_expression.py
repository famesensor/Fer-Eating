import os
import random
import shutil

if __name__ == "__main__":

    folder_list = [
        "neutral",
        "anger",
        "contempt",
        "disgust",
        "fear",
        "happy",
        "sadness",
        "surprise"
    ]

    # init number of files for validation
    neutral = 883
    anger = 162
    contempt = 65
    disgust = 212
    fear = 90
    happy = 248
    sadness = 101
    surprise = 299

    for folder_name in folder_list:
        source = '../dataset/expression/train/' + folder_name
        destination = '../dataset/expression/validate/' + folder_name
        files = os.listdir(source)
        number_of_files = eval(folder_name)

        for file_name in random.sample(files, number_of_files):
            shutil.move(os.path.join(source, file_name), destination)
