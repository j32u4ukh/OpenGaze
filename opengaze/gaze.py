from typing import List
import cv2
import numpy as np
import math
from functools import total_ordering

from coordinate import Point, Vector
from oval import Oval, findOvalIntersection, getOvalPointDistance
from utils import findRadius, findRatio


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
    def __init__(self, height, width, radius: float, n_zone: int):
        self.dst_height = height
        self.dst_width = width
        self.radius = radius
        self.n_zone = n_zone

        # 取 dst 的對角線(最長的長度)
        self.dst_diagonal = math.sqrt(self.dst_height**2 + self.dst_width**2)
        self.dst_rate = findRatio(S=self.dst_diagonal, a=self.radius, n=self.n_zone)
        print(f"diagonal: {self.dst_diagonal}, dst_rate: {self.dst_rate}")
        self.use_dst_ratio = False

        # 預設劃分成三個區段
        self.root_rate = math.sqrt(self.n_zone)
        self.dst_a = self.dst_width / 2 / self.root_rate
        self.dst_b = self.dst_height / 2 / self.root_rate
        self.src_a = None
        self.src_b = None

        self.coord = [0, 0, 0, 0]
        self.zones = {}

        self.img = None
        self.src_height = 0
        self.src_width = 0
        self.channel = 0
        self.distance = 1e-7

    def useDstRatio(self, use: bool):
        self.use_dst_ratio = use

    def initCircleZone(self, src_diagonal: float):
        dst_rate = self.dst_rate if self.use_dst_ratio else 1
        src_rate = findRatio(S=src_diagonal, a=self.radius, n=self.n_zone)
        boundary_list = []
        distance_list = []
        dst_boundary, src_boundary = 0, 0
        n_zone = self.n_zone - 1
        for zone in range(n_zone):
            dst_distance = self.radius * (dst_rate**zone)
            src_distance = self.radius * (src_rate**zone)
            dst_boundary += dst_distance
            src_boundary += src_distance
            boundary_list.append(DstSrc(dst_boundary, src_boundary))
            distance_list.append(DstSrc(dst_distance, src_distance))
            
        boundary_list.append(DstSrc(self.dst_diagonal, src_diagonal))
        distance_list.append(DstSrc(self.dst_diagonal - dst_boundary, src_diagonal - src_boundary))
        
        return self.n_zone, boundary_list, distance_list

    def initOvalZone(self, dst_center: Point, src_center: Point):
        self.src_a = self.src_width / 2 / self.n_zone
        self.src_b = self.src_height / 2 / self.n_zone

        boundary_list = [DstSrc(1, 1)]
        distance_list = [DstSrc(1, 1)]
        dst_ovals = [Oval(center=dst_center, a=self.dst_a, b=self.dst_b)]
        src_ovals = [Oval(center=src_center, a=self.src_a, b=self.src_b)]
        dst_boundary, src_boundary = 0, 0
        n_zone = self.n_zone + 1
        
        for zone in range(1, n_zone):
            dst_boundary = zone + 1
            src_boundary = dst_boundary**2
            src_distance = src_boundary - boundary_list[zone - 1].src
            boundary_list.append(DstSrc(dst_boundary, src_boundary))
            distance_list.append(DstSrc(1, src_distance))

            dst_rate = math.sqrt(dst_boundary)
            dst_a = self.dst_a * dst_rate
            dst_b = self.dst_b * dst_rate

            src_rate = dst_boundary
            src_a = self.src_a * src_rate
            src_b = self.src_b * src_rate

            dst_ovals.append(Oval(center=dst_center, a=dst_a, b=dst_b))
            src_ovals.append(Oval(center=src_center, a=src_a, b=src_b))
        
        self.n_zone = len(boundary_list)
        return self.n_zone, boundary_list, dst_ovals, src_ovals
    
    # 將 src 座標轉換成 dst 座標
    def translateCircle(self, src_center: Point, dst_center: Point, src_pivot: Point, n_zone: int, 
                        boundary_list: List[DstSrc], distance_list: List[DstSrc]) -> tuple[Point, float]:
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

            # 根據座標根據所屬區塊的長度, 將長度做校正
            distance = distance / float(distance_list[zone].src)
            distance = distance * distance_list[zone].dst

            # 若 zone 不為 0, 將座標根據所屬區塊的初始座標值做校正
            if zone != 0:
                distance += boundary_list[zone - 1].dst
            
            vector = vector.norm()
            vector.multiply(distance)            
            point = dst_center.translate(vector)
            dst_w = max(0, min(int(round(point.x)), self.dst_width - 1))
            dst_h = max(0, min(int(round(point.y)), self.dst_height - 1))
            pixel = Point(dst_w, dst_h)
            w_distance = point.computeDistance(pixel)
            weight = 1 / (w_distance + self.distance)
            # weight = gaussian_weight(w_distance)

            return pixel, weight
        except Exception as e:
            print(f"translateCircle Exception: {e}\nsrc_pivot: {src_pivot}, zone: {zone}, vector: {vector}, distance: {distance}")
            return src_pivot, 1
    
    # 將 src 座標轉換成 dst 座標
    def translateOval(self, src_center: Point, dst_center: Point, src_pivot: Point, n_zone: int, 
                      boundary_list: List[DstSrc], dst_ovals: List[Oval], src_ovals: List[Oval]) -> tuple[int, Point, float]:
        vector = Vector(src_center, src_pivot)
        distance = vector.getLength()

        try:
            # 代入橢圓公式，計算橢圓距離，區辨當前屬於第幾個 zone
            elliptic_distance = src_ovals[0].getElliptic(point=src_pivot)

            zone = 0
            for zone in range(n_zone):
                if elliptic_distance <= boundary_list[zone].src:
                    break

            if not self.zones.__contains__(zone):
                self.zones[zone] = 0

            self.zones[zone] += 1

            radius = findRadius(src_center, src_pivot)

            if zone == 0:
                src_inside_point = src_center
            else:
                src_inside_point = src_ovals[zone - 1].getByRadius(radius=radius)
                vector = Vector(src_inside_point, src_pivot)
                distance = vector.getLength()

            src_outside_point = src_ovals[zone].getByRadius(radius=radius)         
            src_gap = src_inside_point.computeDistance(src_outside_point)

            distance /= src_gap

            if zone == 0:
                dst_inside_point = dst_center
            else:
                dst_inside_point = dst_ovals[zone - 1].getByRadius(radius=radius)
            
            dst_outside_point = dst_ovals[zone].getByRadius(radius=radius)
            dst_gap = dst_inside_point.computeDistance(dst_outside_point)

            if dst_outside_point.x > 0:
                if dst_outside_point.y > 0:
                    self.coord[0] += 1
                else:
                    self.coord[3] += 1
            else:                
                if dst_outside_point.y > 0:
                    self.coord[1] += 1
                else:
                    self.coord[2] += 1

            # 根據座標根據所屬區塊的長度, 將長度做校正
            distance *= dst_gap
            
            # 取得單位向量
            vector = vector.norm()

            # 單位向量向量乘以修正後的長度
            vector.multiply(distance)

            # 取得投影後的目標點            
            point = dst_inside_point.translate(vector)

            dst_w = max(0, min(int(round(point.x)), self.dst_width - 1))
            dst_h = max(0, min(int(round(point.y)), self.dst_height - 1))
            pixel = Point(dst_w, dst_h)
            
            w_distance = point.computeDistance(pixel)
            weight = 1 / (w_distance + self.distance)
            return zone, pixel, weight
        except ZeroDivisionError as zde:
            print(f"division by zero, ZeroDivisionError: {zde}, zone: {zone}, src_center: {src_center}, src_pivot: {src_pivot}\nsrc_inside_point: {src_inside_point}, src_outside_point: {src_outside_point}")
        except Exception as e:
            print(f"translateOval Exception: {e}\nsrc_pivot: {src_pivot}, zone: {zone}, vector: {vector}, distance: {distance}")
            return 0, src_pivot, 1

    # 輸出全壓縮圖像，即沒有注視哪一處，全局均勻壓縮
    def basic(self, img: cv2.typing.MatLike):
        self.img = img.copy()
        # shape: (height, width, channel)
        shape = img.shape
        self.src_height = shape[0]
        self.src_width = shape[1]
        self.channel = shape[2]
        basic_img = cv2.resize(img, (self.dst_width, self.dst_height), interpolation=cv2.INTER_CUBIC)
        return basic_img
    
    def gazeCircle(self, x: float = 0, y: float = 0):     
        src_center = Point(self.src_width * x, self.src_height * y )
        dst_center = Point(self.dst_width * x, self.dst_height * y)
        print(f"src_center: {src_center}, dst_center: {dst_center}")

        weights = np.zeros((self.dst_height, self.dst_width, 1), dtype=np.float32)
        values = np.zeros((self.dst_height, self.dst_width, self.channel), dtype=np.float32)
        dst = np.zeros((self.dst_height, self.dst_width, self.channel), dtype=np.uint8)

        src_diagonal = math.sqrt(self.src_height**2 + self.src_width**2)
        zone, boundary_list, distance_list = self.initCircleZone(src_diagonal)
        dst_h, dst_w = 0, 0

        try:
            for h in range(self.src_height):
                for w in range(self.src_width):
                    pivot = Point(w, h)
                    dst_point, weight = self.translateCircle(src_center, dst_center, pivot, zone, boundary_list, distance_list)
                    
                    dst_w = dst_point.x
                    dst_h = dst_point.y
                    color = self.img[h, w]

                    # if (45 <= w and w <= 50) and (45 <= h and h <= 50):
                    #     print(f"pivot: {pivot}, dst_point: {dst_point}, color: {color}")

                    values[dst_h, dst_w] += color * weight
                    weights[dst_h, dst_w] += weight

            # values /= weights
            values /= np.maximum(weights, 1)

            for h in range(self.dst_height):
                for w in range(self.dst_width):
                    color = values[h, w].reshape((1, 1, self.channel))
                    # dst[h, w] = np.round(color).astype(np.uint8)
                    dst[h, w] = np.clip(np.round(color), 0, 255).astype(np.uint8)
        except Exception as e:
            print(e)
            print(f"dst_h: {dst_h}, dst_w: {dst_w}. values: {values.shape}, weights: {weights.shape}")

        return dst
    
    def gazeOval(self, x: float = 0, y: float = 0):     
        src_center = Point(self.src_width * x, self.src_height * y )
        dst_center = Point(self.dst_width * x, self.dst_height * y)
        print(f"src_center: {src_center}, dst_center: {dst_center}")

        weights = np.zeros((self.dst_height, self.dst_width, 1), dtype=np.float32)
        values = np.zeros((self.dst_height, self.dst_width, self.channel), dtype=np.float32)
        dst = np.zeros((self.dst_height, self.dst_width, self.channel), dtype=np.uint8)

        n_zone, boundary_list, dst_ovals, src_ovals = self.initOvalZone(dst_center, src_center)
        # print(f"n_zone: {n_zone}")
        # print(f"boundary_list: {boundary_list}")
        # print(f"dst_ovals: {dst_ovals}")
        # print(f"src_ovals: {src_ovals}")
        dst_h, dst_w = 0, 0

        try:
            for h in range(self.src_height):
                for w in range(self.src_width):
                    pivot = Point(w, h)
                    zone, dst_point, weight = self.translateOval(src_center, dst_center, pivot, n_zone, boundary_list, dst_ovals, src_ovals)
                    
                    dst_w = dst_point.x
                    dst_h = dst_point.y
                    color = self.img[h, w]

                    # if (45 <= w and w <= 50) and (45 <= h and h <= 50):
                    #     print(f"pivot: {pivot}, dst_point: {dst_point}, color: {color}")

                    # if zone == 1:
                    #     values[dst_h, dst_w] = np.array([255, 0, 0])
                    #     weights[dst_h, dst_w] = 1
                    # else:
                    #     values[dst_h, dst_w] += color * weight
                    #     weights[dst_h, dst_w] += weight

                    values[dst_h, dst_w] += color * weight
                    weights[dst_h, dst_w] += weight

            values /= np.maximum(weights, 1)

            for h in range(self.dst_height):
                for w in range(self.dst_width):
                    color = values[h, w].reshape((1, 1, self.channel))
                    dst[h, w] = np.clip(np.round(color), 0, 255).astype(np.uint8)
        except Exception as e:
            print(e)
            print(f"dst_h: {dst_h}, dst_w: {dst_w}. values: {values.shape}, weights: {weights.shape}")

        print(f"coord: {self.coord}")
        print(f"zones: {self.zones}")
        return dst