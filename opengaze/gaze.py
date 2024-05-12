import cv2
import numpy as np
import math

class OpenGaze:
    def __init__(self, height, width, radius: float, distance: float = 1.0):
        self.HEIGHT = height
        self.WIDTH = width

        self.img = None
        self.height = 0
        self.width = 0
        self.channel = 0
        self.distance = distance

        self.radius = radius
        self.n_x_zone = self.countZone(width)
        self.n_y_zone = self.countZone(height)
        self.zone_rate = math.sqrt(2)
        self.zone_boundary = [radius]
        self.zone_distance = [radius]

    def countZone(self, length):
        result = float(length) / float(self.radius)
        n = int(result)

        if result > n:
            return n + 1
        else:
            return n

    def getDistanceX(self, cx: float, x: float) -> float:
        return math.sqrt((x - cx)**2)
    
    def getZone(self, distance: float) -> int:  
        n_boundary = len(self.zone_boundary)
        zone = 0
        r = self.zone_boundary[zone]
        while distance > r:
            zone += 1
            if zone >= n_boundary:
                zone_distance = self.radius*(self.zone_rate**zone)
                self.zone_distance.append(zone_distance)
                self.zone_boundary.append(r+zone_distance)           
            r = self.zone_boundary[zone]
        return zone
    
    def translate(self, x: float, y: float, zone: int) -> float:
        if zone == 0:
            zone_boundary = 0
        else:
            zone_boundary = self.zone_boundary[zone-1]

        # 將 x 轉換為所屬區域的 0 ~ 1
        x = (x-zone_boundary)/self.zone_distance[zone]
        y = (y-zone_boundary)/self.zone_distance[zone]
        return x, y

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
        xi = x*self.width
        yi = y*self.height
        X = None
        Y = None
        count = 0

        for h in range(self.height):
            # 將 h 從原始像素座標，轉換為 0 ~ 1 之間的數值
            hf = float(h) / float(self.height)
            dst_h = math.floor(hf*self.HEIGHT)

            for w in range(self.width):
                # 將 w 從原始像素座標，轉換為 0 ~ 1 之間的數值
                wf = float(w) / float(self.width)
                dst_w = math.floor(wf*self.WIDTH)
                color = self.img[h, w]

                # 計算和注視點的距離
                distance = self.getDistance(xi, yi, w, h)
                zone = self.getZone(distance=distance)

                d_rate = distance / self.radius

                # 注視區域內，直接使用原本的像素
                if d_rate < 1.0:
                    # values[dst_h, dst_w] = color
                    # values[dst_h, dst_w] = np.zeros((1, 1, self.channel), dtype=np.float32)
                    values[dst_h, dst_w] = np.array([[[0, 0, 255]]])
                    weights[dst_h, dst_w] = 1.0
                    
                    if X is None:
                        X = dst_w
                        Y = dst_h
                        count = 1
                    elif dst_w == X and dst_h == Y:
                        count += 1
                elif d_rate < 2.0:
                    values[dst_h, dst_w] = np.array([[[255, 0, 0]]])
                    weights[dst_h, dst_w] = 1.0
                elif d_rate < 3.0:
                    values[dst_h, dst_w] = np.array([[[0, 255, 0]]])
                    weights[dst_h, dst_w] = 1.0
                elif d_rate < 4.0:
                    values[dst_h, dst_w] = np.array([[[0, 0, 255]]])
                    weights[dst_h, dst_w] = 1.0
                else:
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