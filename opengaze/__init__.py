import math
import cv2
import numpy as np

from gaze import OpenGaze
from utils import Point, Vector, computeSn, findRatio

"""
OpenCV 的順序為 (width, height), numpy 呈現的順序為 (height, width, 3)
"""

# 讀取圖片
img = cv2.imread("./data/miko.png")
shape = img.shape
print(f"img.shape: {img.shape}")
src_height = shape[0]
src_width = shape[1]

height = int(src_height/3)
width = int(src_width/3)
radius = 30
n_zone = 16
og = OpenGaze(height=height, width=width, radius=radius, n_zone=n_zone)

basic_img = og.basic(img)
print(f"basic_img.shape: {basic_img.shape}")

dst = og.gazeCircle(0.5, 0.5)
print(f"dst.shape: {dst.shape}")

cv2.imwrite(f"./data/gaze-({height}, {width})-r({radius})-{n_zone}.png", dst)
# cv2.imshow("PON", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
