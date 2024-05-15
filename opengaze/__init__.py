import math
import cv2
import numpy as np

from gaze import OpenGaze

"""
OpenCV 的順序為 (width, height), numpy 呈現的順序為 (height, width, 3)
"""

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
class Vector:
    def __init__(self, p1: Point, p2: Point):
        self.m = None if p1.x == p2.x else (p2.y - p1.y) / (p2.x - p1.x)
        self.c = p1.y if self.m is None else p1.y - self.m * p1.x

    def getYByX(self, x):
        if self.m is None:
            return None
        else:
            return self.m * x + self.c
    
    def getXByY(self, y):
        if self.m == 0:
            return None
        elif self.m is None:
            return y
        else:
            return (y - self.c) / self.m

# 計算兩向量交點   
def crossPoint(v1: Vector, v2: Vector) -> Point:
    # 求兩條直線的交點座標
    if v1.m == v2.m:
        # 如果兩條直線平行，則返回 None
        return None
    elif v1.m is None:
        # 如果其中一條直線為垂直線，則交點的 x 座標為 v1 的 x 座標
        x = v1.c
        y = v2.getYByX(x)
    elif v2.m is None:
        # 如果其中一條直線為垂直線，則交點的 x 座標為 v2 的 x 座標
        x = v2.c
        y = v1.getYByX(x)
    else:
        # 通常情況下，使用方程式解交點的 x 座標
        x = (v2.c - v1.c) / (v1.m - v2.m)
        y = v1.getYByX(x)
    
    return Point(x, y)

# 測試
p1 = Point(1, 1)
p2 = Point(4, 5)
p3 = Point(2, 3)
p4 = Point(6, 7)

v1 = Vector(p1, p2)
v2 = Vector(p3, p4)

intersection = crossPoint(v1, v2)
if intersection:
    print("兩條直線的交點為: ({}, {})".format(intersection.x, intersection.y))
else:
    print("兩條直線平行，沒有交點。")

# # 讀取圖片
# img = cv2.imread("./data/miko.png")
# shape = img.shape
# print(f"img.shape: {img.shape}")
# src_height = shape[0]
# src_width = shape[1]
# height = int(src_height/3)
# width = int(src_width/3)


# og = OpenGaze(height=height, width=width, radius=40, distance=1)
# zone_rate = 1.1

# zone, boundary_list, distance_list = og.initDistanceZone(rate=zone_rate)
# print(f"zone: {zone}")
# print(f"boundary_list({len(boundary_list)}): {boundary_list}")
# print(f"distance_list({len(distance_list)}): {distance_list}")

# basic_img = og.basic(img)
# print(f"basic_img.shape: {basic_img.shape}")

# dst = og.gaze(0.5, 0.5)
# print(f"dst.shape: {dst.shape}")

# dst_pivot = og.translate(1920, w_zone, w_boundary_list, w_distance_list)
# print(f"dst_pivot: {dst_pivot}")

# cv2.imwrite("./data/gaze.png", dst)
# cv2.imshow("PON", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


