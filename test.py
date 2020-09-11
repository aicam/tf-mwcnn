import numpy as np
import matplotlib.pyplot as plt
from select_area import *
import cv2
filename = './results/Rec--000098_AmpAll/3.png'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
criteria = (cv2.TERM_CRITERIA_EPS, )
# Threshold for an optimal value, it may vary depending on the image.
corner_matrix = dst>0.05*dst.max()
corner_matrix = corner_matrix.astype(np.int32)
cells, start_point, end_point = corner_matrix_process(corner_matrix)
# img = cv2.rectangle(img, start_point, end_point, (0, 0, 255), 2)
# img[cells[:,0], cells[:,1]]=[0,0,255]
# font                   = cv2.FONT_HERSHEY_SIMPLEX
# bottomLeftCornerOfText = (np.min(cells , axis=0)[1], np.min(cells, axis=0)[0] - 5)
# print(bottomLeftCornerOfText)
# fontScale              = 0.6
# fontColor              = (0,0,255)
# lineType               = 2
# cv2.putText(img,'82%',
#     bottomLeftCornerOfText,
#     font,
#     fontScale,
#     fontColor,
#     lineType)
plt.imshow(img)
plt.show()
# cv2.imshow('dst',img)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()