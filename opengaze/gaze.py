from typing import List
import cv2
import numpy as np
import math
from functools import total_ordering

from utils import Point, Vector, computeDistance, findRatio, getUnitVector, modifyPoint

# total_ordering: 使得我可以只定義 __eq__ 和 __gt__ 就可進行完整的比較
# https://python3-cookbook.readthedocs.io/zh_CN/latest/c08/p24_making_classes_support_comparison_operations.html
@total_ordering
class DstSrc:
    def __init__(self, dst, src):
        self.dst = dst
        self.src = src

    def __eq__(self, other):
        return self.dst == other.dst

    def __gt__(self, other):
        return self.dst > other.dst
    
    def __repr__(self) -> str:
        return f"({self.dst}, {self.src})"
    
    __str__ = __repr__


class OpenGaze:
    def __init__(self, height, width, radius: float, distance: float = 1.0):
        self.dst_height = height
        self.dst_width = width
        self.radius = radius
        self.zone_rate = math.sqrt(2)
        # 取 dst 的對角線(最長的長度)
        self.diagonal = math.sqrt(self.dst_height**2 + self.dst_width**2)
        print(f"diagonal: {self.diagonal}")

        self.img = None
        self.height = 0
        self.width = 0
        self.channel = 0
        self.distance = distance

    def initCircleZone(self, src_diagonal: float, dst_diagonal: float):
        n_zone = 10
        self.zone_rate = findRatio(S=src_diagonal, a=self.radius, n=n_zone)
        boundary_list = []
        distance_list = []
        dst_boundary, src_boundary = 0, 0

        zone = 0
        while dst_boundary < self.diagonal:
            dst_boundary += self.radius
            distance = self.radius * (self.zone_rate**zone)
            src_boundary += distance
            boundary_list.append(DstSrc(dst_boundary, src_boundary))
            distance_list.append(DstSrc(self.radius, distance))
            if zone == n_zone - 2:
                break
            zone += 1

        boundary_list.append(DstSrc(dst_diagonal, src_diagonal))
        distance_list.append(DstSrc(dst_diagonal - dst_boundary, src_diagonal - src_boundary))
        
        return zone, boundary_list, distance_list

    def initZone(self, pivot: float, dst_length: int, src_length: int, rate: float):        
        dst_center = pivot * dst_length
        src_center = pivot * src_length
        boundary_list = [DstSrc(dst_center, src_center)]
        self.initLeftZone(dst_center, src_center, rate, boundary_list)
        self.initRightZone(dst_center, src_center, dst_length, rate, boundary_list)

        boundary_list.sort()
        n_zone = len(boundary_list)
        distance_list = []

        prev, curr = 0, 0
        for i in range(1, n_zone):
            prev = boundary_list[i - 1]
            curr = boundary_list[i]
            dst_distance = curr.dst - prev.dst
            src_distance = curr.src - prev.src
            distance_list.append(DstSrc(dst_distance, src_distance))
        
        if curr.dst < dst_length:
            distance_list.append(DstSrc(dst_length - curr.dst, src_length - curr.src))

        boundary_list.pop(0)

        if distance_list[-1].dst < self.radius:
            distance_list[-2].dst += distance_list[-1].dst
            distance_list[-2].src += distance_list[-1].src
            distance_list.pop()
            boundary_list.pop()

        if distance_list[0].dst < self.radius:
            distance_list[1].dst += distance_list[0].dst
            distance_list[1].src += distance_list[0].src
            distance_list.pop(0)
            boundary_list.pop(0)
        
        n_zone = len(boundary_list)
        return n_zone, boundary_list, distance_list
    
    def initLeftZone(self, dst_boundary: float, src_boundary: float, rate: float, boundary_list: List[float]):
        i = 0
        while dst_boundary > 0:
            dst_boundary -= self.radius
            src_boundary -= self.radius * (rate**i)
            if dst_boundary > 0:
                boundary_list.append(DstSrc(dst_boundary, src_boundary))
                i += 1
            else:
                boundary_list.append(DstSrc(0, 0))
    
    def initRightZone(self, dst_boundary: float, src_boundary: float, length: int, rate: float, boundary_list: List[float]):
        i = 0
        while dst_boundary < length:
            dst_boundary += self.radius
            src_boundary += self.radius * (rate**i)
            if dst_boundary < length:
                boundary_list.append(DstSrc(dst_boundary, src_boundary))
                i += 1
            # else:
            #     boundary_list.append(length - 1)
    
    # 將 src 座標轉換成 dst 座標
    # boundary_list: [(50.0, 100.0), (100.0, 150.0), (150.0, 200.0)]
    # distance_list: [(50.0, 100.0), (50.0, 50.0), (50.0, 50.0), (50.0, 100.0)]
    def translate(self, src_pivot: int, n_zone: int, boundary_list: List[DstSrc], distance_list: List[DstSrc]) -> int:
        zone = 0
        not_found = True
        for zone in range(n_zone):
            if src_pivot <= boundary_list[zone].src:
                not_found = False
                break
        if not_found:
            zone = n_zone

        pivot = src_pivot
        # 若 zone 不為 0, 將座標根據所屬區塊的初始座標值做校正
        if zone != 0:
            pivot -= boundary_list[zone - 1].src
        # 根據座標根據所屬區塊的長度, 將長度做校正
        pivot = float(pivot) / distance_list[zone].src
        pivot = pivot * distance_list[zone].dst
        # 若 zone 不為 0, 將座標根據所屬區塊的初始座標值做校正
        if zone != 0:
            pivot += boundary_list[zone - 1].dst
        return int(round(pivot))
    
    def translateCircle(self, src_center: Point, dst_center: Point, src_pivot: Point, n_zone: int, boundary_list: List[DstSrc], distance_list: List[DstSrc]) -> tuple[Point, float]:
        try:
            vector = Vector(src_center, src_pivot)
            distance = vector.getLength()

            zone = 0
            for zone in range(n_zone):
                if distance <= boundary_list[zone].src:
                    break

            # 若 zone 不為 0, 將座標根據所屬區塊的初始座標值做校正
            if zone != 0:
                distance -= boundary_list[zone - 1].src

            # weight = 1 / (distance + self.distance)
            weight = 1

            # 根據座標根據所屬區塊的長度, 將長度做校正
            distance = distance / float(distance_list[zone].src)
            distance = distance * distance_list[zone].dst

            # 若 zone 不為 0, 將座標根據所屬區塊的初始座標值做校正
            if zone != 0:
                distance += boundary_list[zone - 1].dst
            
            vector = getUnitVector(vector)
            vector.multiply(distance)
            return modifyPoint(dst_center, vector), weight
        except Exception as e:
            print(f"translateCircle Exception: {e}\nsrc_pivot: {src_pivot}, zone: {zone}, vector: {vector}, distance: {distance}")
            return src_pivot, 1

    # 輸出全壓縮圖像，即沒有注視哪一處，全局均勻壓縮
    def basic(self, img: cv2.typing.MatLike):
        self.img = img.copy()
        # shape: (height, width, channel)
        shape = img.shape
        self.height = shape[0]
        self.width = shape[1]
        self.channel = shape[2]
        basic_img = cv2.resize(img, (self.dst_width, self.dst_height), interpolation=cv2.INTER_CUBIC)
        return basic_img

    def gaze(self, x: float = 0, y: float = 0): 
        weights = np.ones((self.dst_height, self.dst_width, 1), dtype=np.float32)
        values = np.zeros((self.dst_height, self.dst_width, self.channel), dtype=np.float32)
        dst = np.zeros((self.dst_height, self.dst_width, self.channel), dtype=np.uint8)

        w_zone_number, w_boundary_list, w_distance_list = self.initZone(x, self.dst_width, self.width, self.zone_rate)
        h_zone_number, h_boundary_list, h_distance_list = self.initZone(y, self.dst_height, self.height, self.zone_rate)

        X, Y = None, None
        count = 0
        dst_h, dst_w = 0, 0

        try:
            for h in range(self.height):
                dst_h = self.translate(h, h_zone_number, h_boundary_list, h_distance_list)
                dst_h = min(dst_h, self.dst_height - 1)
                for w in range(self.width):
                    dst_w = self.translate(w, w_zone_number, w_boundary_list, w_distance_list)
                    dst_w = min(dst_w, self.dst_width - 1)

                    color = self.img[h, w]
                    values[dst_h, dst_w] += color
                    weights[dst_h, dst_w] += 1

            values /= weights
        
            for h in range(self.dst_height):
                for w in range(self.dst_width):
                    color = values[h, w].reshape((1, 1, self.channel))
                    dst[h, w] = np.round(color).astype(np.uint8)
        except Exception as e:
            print(e)
            print(f"dst_h: {dst_h}, dst_w: {dst_w}. values: {values.shape}, weights: {weights.shape}")

        return dst
    
    def gazeCircle(self, x: float = 0, y: float = 0):     
        src_center = Point(self.width * x, self.height * y )
        dst_center = Point(self.dst_width * x, self.dst_height * y)
        print(f"src_center: {src_center}, dst_center: {dst_center}")

        weights = np.ones((self.dst_height, self.dst_width, 1), dtype=np.float32)
        values = np.zeros((self.dst_height, self.dst_width, self.channel), dtype=np.float32)
        dst = np.zeros((self.dst_height, self.dst_width, self.channel), dtype=np.uint8)

        src_diagonal = math.sqrt(self.height**2 + self.width**2)
        dst_diagonal = math.sqrt(self.dst_height**2 + self.dst_width**2)
        zone, boundary_list, distance_list = self.initCircleZone(src_diagonal, dst_diagonal)

        W, H = None, None
        count = 0
        dst_h, dst_w = 0, 0

        try:
            for h in range(self.height):
                for w in range(self.width):
                    pivot = Point(w, h)
                    dst_point, weight = self.translateCircle(src_center, dst_center, pivot, zone, boundary_list, distance_list)
                    
                    dst_w = max(0, min(int(round(dst_point.x)), self.dst_width - 1))
                    dst_h = max(0, min(int(round(dst_point.y)), self.dst_height - 1))
                    color = self.img[h, w]

                    # if (45 <= w and w <= 50) and (45 <= h and h <= 50):
                    #     print(f"pivot: {pivot}, dst_point: {dst_point}, color: {color}")

                    values[dst_h, dst_w] += color
                    weights[dst_h, dst_w] += weight

            values /= weights
        
            for h in range(self.dst_height):
                for w in range(self.dst_width):
                    color = values[h, w].reshape((1, 1, self.channel))
                    dst[h, w] = np.round(color).astype(np.uint8)
        except Exception as e:
            print(e)
            print(f"dst_h: {dst_h}, dst_w: {dst_w}. values: {values.shape}, weights: {weights.shape}")

        return dst