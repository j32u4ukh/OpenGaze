from typing import List
import cv2
import numpy as np
import math
from functools import total_ordering

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
        self.zone_rate = math.sqrt(2)
        self.HEIGHT = height
        self.WIDTH = width
        self.radius = radius

        self.img = None
        self.height = 0
        self.width = 0
        self.channel = 0
        self.distance = distance

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
        return pivot

    # 輸出全壓縮圖像，即沒有注視哪一處，全局均勻壓縮
    def basic(self, img: cv2.typing.MatLike):
        self.img = img.copy()
        # shape: (height, width, channel)
        shape = img.shape
        self.height = shape[0]
        self.width = shape[1]
        self.channel = shape[2]
        basic_img = cv2.resize(img, (self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_CUBIC)
        return basic_img

    def gaze(self, x: float = 0, y: float = 0): 
        weights = np.zeros((self.HEIGHT, self.WIDTH, 1), dtype=np.float32)
        values = np.zeros((self.HEIGHT, self.WIDTH, self.channel), dtype=np.float32)

        w_zone_number, w_boundary_list, w_distance_list = self.initZone(x, self.WIDTH, self.width, self.zone_rate)
        h_zone_number, h_boundary_list, h_distance_list = self.initZone(y, self.HEIGHT, self.height, self.zone_rate)

        X = None
        Y = None
        count = 0

        for h in range(self.height):
            dst_h = self.translate(h, h_zone_number, h_boundary_list, h_distance_list)
            for w in range(self.width):
                dst_w = self.translate(w, w_zone_number, w_boundary_list, w_distance_list)

                color = self.img[h, w]
                values[dst_h, dst_w] += color
                weights[dst_h, dst_w] += 1
                     

        values /= weights        
        dst = np.zeros((self.HEIGHT, self.WIDTH, self.channel), dtype=np.uint8)
        print(f"count: {count}")
     
        for h in range(self.HEIGHT):
            for w in range(self.WIDTH):
                color = values[h, w].reshape((1, 1, self.channel))
                dst[h, w] = np.round(color).astype(np.uint8)

        return dst