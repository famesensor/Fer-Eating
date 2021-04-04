import os
import random
import shutil

eatSource = '../dataset/behavior/train/eat'
eatDest = '../dataset/behavior/validate/eat'
eatFiles = os.listdir(eatSource)
eatNumberOfFiles = 700

noEatSource = '../dataset/behavior/train/noeat'
noEatDest = '../dataset/behavior/validate/noeat'
noEatFiles = os.listdir(noEatSource)
noEatNumberOfFiles = 1200

for file_name in random.sample(eatFiles, eatNumberOfFiles):
    shutil.move(os.path.join(eatSource, file_name), eatDest)

for file_name in random.sample(noEatFiles, noEatNumberOfFiles):
    shutil.move(os.path.join(noEatSource, file_name), noEatDest)
