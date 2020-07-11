import numpy as np
import matplotlib.pyplot as plt
from os import listdir
# silver spayer, pomad alpha, zinc
from os.path import isfile, join
import time
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
    # test_mwcnn_change(X)
    f.close()
import cv2
filename = 'results/2/3.png'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.1*dst.max()]=[0,0,255]
cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()