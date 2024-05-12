import math
import cv2
import numpy as np

from gaze import OpenGaze

"""
OpenCV 的順序為 (width, height), numpy 呈現的順序為 (height, width, 3)
"""

# # 讀取圖片
# img = cv2.imread("./data/miko.png")
# shape = img.shape
# print(f"img.shape: {img.shape}")
# src_height = shape[0]
# src_width = shape[1]
# height = int(src_height/3)
# width = int(src_width/3)

# og = OpenGaze(height=height, width=width, radius=40, distance=1)
# zone_rate = math.sqrt(2)

# x = 1 + math.sqrt(2)
# print(f"x: {x}")
# distance = og.getDistanceX(0, x)
# zone = og.getZoneX(distance=distance)
# print(f"distance: {distance}, zone: {zone}")
# translate_x = og.translate(x, zone)
# print(f"translate_x: {translate_x}")
# basic_img = og.basic(img)
# print(f"basic_img.shape: {basic_img.shape}")

# dst = og.gaze(0.5, 0.5)
# print(f"dst.shape: {dst.shape}")

# boundary_list, distance_list
# w_zone, w_boundary_list, w_distance_list = og.initZone(0.5, width, src_width, zone_rate)
# print(f"w_boundary_list: {w_boundary_list}\nw_distance_list: {w_distance_list}")
# h_zone, h_boundary_list, h_distance_list = og.initZone(0.5, height, src_height, zone_rate)
# print(f"h_boundary_list: {h_boundary_list}\nh_distance_list: {h_distance_list}")

# dst_pivot = og.translate(1920, w_zone, w_boundary_list, w_distance_list)
# print(f"dst_pivot: {dst_pivot}")
# cv2.imwrite("./data/gaze.png", dst)
# cv2.imshow("PON", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



import cv2
import numpy as np

def convex(src_img, raw, effect):
    col, row, channel = raw[:]      # 取得圖片資訊
    cx, cy, r = effect[:]           # 取得凸透鏡的範圍
    output = np.zeros([row, col, channel], dtype = np.uint8)        # 產生空白畫布
    for y in range(row):
        for x in range(col):
            d = ((x - cx) * (x - cx) + (y - cy) * (y - cy)) ** 0.5  # 計算每個點與中心點的距離
            if d <= r:
                nx = int((x - cx) * d / r + cx)        # 根據不同的位置，產生新的 nx，越靠近中心形變越大
                ny = int((y - cy) * d / r + cy)        # 根據不同的位置，產生新的 ny，越靠近中心形變越大
                output[y, x, :] = src_img[ny, nx, :]   # 產生新的圖
            else:
                output[y, x, :] = src_img[y, x, :]     # 如果在半徑範圍之外，原封不動複製過去
    return output

img = cv2.imread("./data/miko.png")
img = convex(img, (1920, 1080, 3), (150, 130, 100))      # 提交參數數值，進行凸透鏡效果
cv2.imshow('oxxostudio', img)
cv2.waitKey(0)
cv2.destroyAllWindows()