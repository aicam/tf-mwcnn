import numpy as np
from os import listdir
from os.path import isfile, join
def get_data(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    X = []

    for file in onlyfiles:
        f = open(mypath + '' + file)
        x_ = []
        for l in f:
            x = np.array([item.rstrip() for item in l.split()])
            x.astype(np.float)
            x_.append(x)
        X.append(np.array(x_).reshape([-1, 64, 80]).astype(np.float32))
        f.close()
