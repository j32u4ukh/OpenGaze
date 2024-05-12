import math
import cv2
import numpy as np

from gaze import OpenGaze

"""
OpenCV 的順序為 (width, height), numpy 呈現的順序為 (height, width, 3)
"""

# 讀取圖片
img = cv2.imread("./data/miko.png")
shape = img.shape
print(f"img.shape: {img.shape}")
height = int(shape[0]/3)
width = int(shape[1]/3)

og = OpenGaze(height=height, width=width, radius=50, distance=1)
zone_rate = math.sqrt(2)

# x = 1 + math.sqrt(2)
# print(f"x: {x}")
# distance = og.getDistanceX(0, x)
# zone = og.getZoneX(distance=distance)
# print(f"distance: {distance}, zone: {zone}")
# translate_x = og.translate(x, zone)
# print(f"translate_x: {translate_x}")
basic_img = og.basic(img)
print(f"basic_img.shape: {basic_img.shape}")

# dst = og.gaze(0.5, 0.2)
# print(f"dst.shape: {dst.shape}")
w_n_zone, w_distance_list, w_cum_distance_list = og.initZone(0.5, 400, 700, zone_rate)
print(f"w_n_zone: {w_n_zone}\nw_distance_list: {w_distance_list}\nw_cum_distance_list: {w_cum_distance_list}")
h_n_zone, h_distance_list, h_cum_distance_list = og.initZone(0.5, 200, 300, zone_rate)
print(f"h_n_zone: {h_n_zone}\nh_distance_list: {h_distance_list}\nh_cum_distance_list: {h_cum_distance_list}")

# cv2.imwrite("./data/gaze.png", dst)
# cv2.imshow("PON", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()