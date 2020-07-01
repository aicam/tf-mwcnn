import numpy as np
import matplotlib.pyplot as plt
datContent = np.array([np.array(i.strip().split()).astype(np.float) for i in open("./dataset/1.dat").readlines()])
im = datContent.reshape([-1, 64, 80])


plt.imshow(im[10])
plt.show()