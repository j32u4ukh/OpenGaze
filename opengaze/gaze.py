from typing import List
import cv2
import numpy as np
import math

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

    def initZone(self, pivot: float, length: int, rate: float):
        center = pivot * length
        boundary_list = [center]
        self.initLeftZone(center, rate, boundary_list)
        self.initRightZone(center, length, rate, boundary_list)
        boundary_list.sort()
        n_zone = len(boundary_list)
        distance_list = []

        prev, curr = 0, 0
        for i in range(1, n_zone):
            prev = boundary_list[i - 1]
            curr = boundary_list[i]
            distance_list.append(curr - prev)
        
        if curr < length:
            distance_list.append(length - curr)

        boundary_list.pop(0)

        if rate == 1.0:
            if distance_list[-1] < self.radius:
                distance_list[-2] += distance_list[-1]
                distance_list.pop()
                boundary_list.pop()

            if distance_list[0] < self.radius:
                distance_list[1] += distance_list[0]
                distance_list.pop(0)
                boundary_list.pop(0)
                
        return n_zone, boundary_list, distance_list
    
    def initLeftZone(self, boundary: float, rate: float, boundary_list: List[float]):
        i = 0
        while boundary > 0:
            boundary -= self.radius * (rate**i)
            if boundary > 0:
                boundary_list.append(boundary)
                i += 1
            else:
                boundary_list.append(0)
    
    def initRightZone(self, boundary: float, length: int, rate: float, boundary_list: List[float]):
        i = 0
        while boundary < length:
            boundary += self.radius * (rate**i)
            if boundary < length:
                boundary_list.append(boundary)
                i += 1
            # else:
            #     boundary_list.append(length - 1)
    
    def getZone(self, distance: float, is_width: bool) -> int:
        is_neg = distance < 0
        distance = math.fabs(distance)
        if is_width:
            for zone in range(self.w_zone_number):
                if distance <= self.w_cum_distance_list[zone]:
                    return zone
            return self.w_zone_number - 1
        else:
            for zone in range(self.h_zone_number):
                if distance <= self.h_cum_distance_list[zone]:
                    return zone
            return self.h_zone_number - 1
    
    def translate(self, v: float, zone: int, is_width: bool) -> float:
        if zone == 0:
            zone_boundary = 0
        else:
            zone_boundary = self.zone_boundary[zone-1]

        # 將座標轉換為所屬區域的 0 ~ 1
        v = (v-zone_boundary)/self.zone_distance[zone]
        return v

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
    
    def setRadius(self, radius: float):
        self.radius = radius
        self.zone_boundary = [radius]
        self.zone_distance = [radius]
    
    def getDistance(self, cx: float = 0, cy: float = 0, x: float = 0, y: float = 0) -> float:
        distance = math.sqrt((x - cx)**2 + (y - cy)**2)
        return distance
    
    def getWeight(self, distance: float = 0) -> float:
        return 1.0 / (self.distance + distance)
    
    def gaze(self, x: float = 0, y: float = 0): 
        weights = np.zeros((self.HEIGHT, self.WIDTH, 1), dtype=np.float32)
        values = np.zeros((self.HEIGHT, self.WIDTH, self.channel), dtype=np.float32)
        w_i = x*self.width
        h_i = y*self.height
        X = None
        Y = None
        count = 0

        for h in range(self.height):
            # 將 h 從原始像素座標，轉換為 0 ~ 1 之間的數值
            hf = float(h) / float(self.height)
            # dst_h = math.floor(hf*self.HEIGHT)

            for w in range(self.width):
                # 將 w 從原始像素座標，轉換為 0 ~ 1 之間的數值
                wf = float(w) / float(self.width)
                # dst_w = math.floor(wf*self.WIDTH)
                color = self.img[h, w]

                # 計算和注視點的距離
                w_distance = w_i - w
                h_distance = h_i - h

                # 計算 z 分數, 以及縮放比例
                w_zone = self.getZone(distance=w_distance)
                h_zone = self.getZone(distance=h_distance)
                w_translate = self.translate(w, w_zone)
                h_translate = self.translate(w, h_zone)
                
                dst_w = math.floor((w_zone + w_translate) * self.radius)
                dst_h = math.floor((h_zone + h_translate) * self.radius)

                if w_zone == 0 and h_zone == 0:
                    values[dst_h, dst_w] = color
                    # values[dst_h, dst_w] = np.zeros((1, 1, self.channel), dtype=np.float32)
                    # values[dst_h, dst_w] = np.array([[[0, 0, 255]]])
                    weights[dst_h, dst_w] = 1.0
                else:
                    distance = self.getDistance(w_i, h_i, h, w)
                    weight = self.getWeight(distance)
                    values[dst_h, dst_w] += color * weight
                    weights[dst_h, dst_w] += weight  

        values /= weights        
        dst = np.zeros((self.HEIGHT, self.WIDTH, self.channel), dtype=np.uint8)
        print(f"count: {count}")
     
        for h in range(self.HEIGHT):
            for w in range(self.WIDTH):
                color = values[h, w].reshape((1, 1, self.channel))
                dst[h, w] = np.round(color).astype(np.uint8)

        return dst