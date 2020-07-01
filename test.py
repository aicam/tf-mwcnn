from mwcnn_test import *
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
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
print(X[0][1][:,4])
# plt.imshow(X[0][1][:,3:3])
# plt.show()
# plt.imshow(X[60])
# plt.plot()
# test_mwcnn_change(X)
# for i in range(10):
#     test_mwcnn_change(X[0][i])
#     time.sleep(2)
