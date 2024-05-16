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

og = OpenGaze(height=height, width=width, radius=40, distance=1e-7)
# diagonal = math.sqrt(src_height**2 + src_width**2)
# zone, boundary_list, distance_list = og.initDistanceZone(src_diagonal=diagonal)
# print(f"zone: {zone}, zone_rate: {og.zone_rate}")
# print(f"boundary_list({len(boundary_list)}): {boundary_list}")
# print(f"distance_list({len(distance_list)}): {distance_list}")

basic_img = og.basic(img)
print(f"basic_img.shape: {basic_img.shape}")

dst = og.gazeCircle(0.5, 0.5)
print(f"dst.shape: {dst.shape}")

cv2.imwrite("./data/gaze.png", dst)
# cv2.imshow("PON", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# vector = Vector(Point(0, 0), Point(1920, 1080))
# length = vector.getLength()
# print(f"length: {length}")
