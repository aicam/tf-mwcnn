import numpy as np
from os import listdir
from os.path import isfile, join


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

# 2 : mean + mean
# 3 : var + std
# 4 : std + std
stds = []

for l in range(len(X[0])):
    im_std = []
    for i in range(0,64,8):
        for j in range(0,80,8):
            im_std.append(np.var(X[0][l][i:i+8,j:j+8]))
    stds.append(im_std)
std_std = []

for i in range(len(stds)):
    std_std.append(np.std(stds[i]))
print(np.argmax(std_std))