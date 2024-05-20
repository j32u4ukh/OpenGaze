import math
import cv2
import numpy as np

from gaze import OpenGaze
from utils import Point, Vector, computeSn, findOvalIntersection, findOvalPointByRadius, findRadius, findRatio, getOvalDistance

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
n_zone = 3
og = OpenGaze(height=height, width=width, radius=radius, n_zone=n_zone)

basic_img = og.basic(img)
print(f"basic_img.shape: {basic_img.shape}")

n_zone, a_list, b_list, boundary_list, distance_list = og.initOvalZone()

# x_float = 0.5
# y_float = 0.5
# src_center = Point(og.src_width * x_float, og.src_height * y_float)
# dst_center = Point(og.dst_width * x_float, og.dst_height * y_float)
# pivot = Point(src_center.x + 5, src_center.y + 10)
# dst_point, weight = og.translateOval(src_center, dst_center, pivot, n_zone, a_list, b_list, boundary_list)
# print(f"dst_point: {dst_point}")
# print(f"weight: {weight}")


dst = og.gazeOval(0.5, 0.5)
print(f"dst.shape: {dst.shape}")

cv2.imwrite(f"./data/gazeoval-({height}, {width})-{n_zone}.png", dst)
# cv2.imshow("PON", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
