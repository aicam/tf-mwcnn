import cv2
import numpy as np

MINI_FILTER_SIZE = 8
MIDI_FILTER_SIZE = 16
KERNEL_SHARPEN = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]])/8.0

def filter_image(X):
    im_mean_mini = []
    for i in range(0, 64, MINI_FILTER_SIZE):
        for j in range(0, 80, MINI_FILTER_SIZE):
            im_mean_mini.append(np.mean(X[i:i + MINI_FILTER_SIZE, j:j + MINI_FILTER_SIZE]))
    im_std_midi = []
    for i in range(0, 64, MIDI_FILTER_SIZE):
        for j in range(0, 80, MIDI_FILTER_SIZE):
            im_std_midi.append(np.mean(X[i:i + MIDI_FILTER_SIZE, j:j + MIDI_FILTER_SIZE]))
    return np.mean(im_mean_mini), np.std(im_std_midi)


def get_batch_filtered(images):
    filter_results = []
    for i in range(len(images)):
        mean_mini, std_midi = filter_image(images[i])
        filter_results.append([mean_mini, std_midi])
    return np.array(filter_results)

# remove low mean of colors
def remove_phase_1(images):
    result_arr = get_batch_filtered(images)
    remove_index = np.argsort(result_arr[:, 0])[:50]
    result_arr = np.delete(result_arr, remove_index, 0)
    return np.argsort(result_arr[:,1])[::-1][:3]


def sharpen_image(image):
    return cv2.filter2D(image, -1, KERNEL_SHARPEN)