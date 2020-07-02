import os
from skimage.transform import resize
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from image_processing import *
import matplotlib.pyplot as plt
mypath = './dataset'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
X = []
Y = []

for file in onlyfiles:
    f = open('./dataset/' + file)
    x_ = []
    for l in f:
        x = np.array([item.rstrip() for item in l.split()])
        x.astype(np.float)
        x_.append(x)
    X.append(np.array(x_).reshape([-1, 64, 80]).astype(np.float32))
    f.close()



for i in range(len(X)):
    os.makedirs('./results/' + onlyfiles[i].replace('.dat', ''))
    best_images = remove_phase_1(X[i])
    for k in best_images:
        try:
            img = resize(X[i][k], (480, 640))
            img = sharpen_image(img)
            plt.imsave('./results/' + onlyfiles[i].replace('.dat', '') + '/' + str(k) + '.png', img)
        except FileExistsError:
            pass