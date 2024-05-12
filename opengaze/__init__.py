import math
import cv2
import numpy as np

from gaze import OpenGaze

"""
OpenCV 的順序為 (width, height), numpy 呈現的順序為 (height, width, 3)
"""

# 讀取圖片
img = cv2.imread("./data/starts.png")
shape = img.shape
print(f"img.shape: {img.shape}")

og = OpenGaze(height=int(shape[0]/3), width=int(shape[1]/3), radius=1, distance=0.5)
x = 1 + math.sqrt(2)
print(f"x: {x}")
distance = og.getDistanceX(0, x)
zone = og.getZoneX(distance=distance)
print(f"distance: {distance}, zone: {zone}")
translate_x = og.translate(x, zone)
print(f"translate_x: {translate_x}")
# basic_img = og.basic(img)
# print(f"basic_img.shape: {basic_img.shape}")

# dst = og.gaze(0.5, 0.2)
# print(f"dst.shape: {dst.shape}")


# cv2.imwrite("./data/gaze.png", dst)
# cv2.imshow("PON", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()