import numpy as np

def filter_image(X):
    im_std = []
    for i in range(0, 64, 8):
        for j in range(0, 80, 8):
            im_std.append(np.var(X[i:i + 8, j:j + 8]))
    return np.std(im_std)

def get_batch_filtered(images):
    filter_results = []
    for image in images:
        filter_results.append(filter_image(image))
    return np.argsort(filter_results)[::-1][:3]

