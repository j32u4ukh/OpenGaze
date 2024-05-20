import math
import numpy as np


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point(x: {self.x}, y: {self.y})"
    
    __str__ = __repr__


class Vector:
    def __init__(self, p1: Point, p2: Point):
        self.x = p2.x - p1.x
        self.y = p2.y - p1.y
        
    def getLength(self):
        return math.sqrt(self.x**2 + self.y**2)
    
    def multiply(self, value: float):
        self.x *= value
        self.y *= value

    def __repr__(self):
        return f"Vector(x: {self.x}, y: {self.y})"
    
    __str__ = __repr__
    
       
def getUnitVector(v: Vector) -> Vector:
    length = v.getLength()
    if length == 0:
        return v
    x = v.x / length
    y = v.y / length
    return Vector(p1=Point(0, 0), p2=Point(x, y))

def modifyPoint(p: Point, v: Vector) -> Point:
    return Point(x = p.x + v.x, y = p.y + v.y)

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
 
def computeDistance(p1: Point, p2: Point) -> float:
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# 計算等比級數前 n 項的總和(a: 首項; r: 公差)
def computeSn(a, r, n):
    return (a * (r**n - 1)) / (r - 1)

def findRatio(S, a, n, tolerance=1e-6):
    # 初始猜測範圍
    low = 1.01  # r 大於 1
    high = 10   # 任意較大的數來保證範圍

    # 二分法迭代
    while (high - low) > tolerance:
        mid = (low + high) / 2
        # Sn = a * (mid**n - 1) / (mid - 1)
        Sn = computeSn(a, mid, n)
        
        if Sn > S:
            high = mid
        else:
            low = mid

    # 返回逼近的公比 r
    return (low + high) / 2


def getOvalPointDistance(a: float, b: float, center: Point, point: Point) -> float:
    return getOvalDistance(a=a, b=b, cx=center.x, cy=center.y, x=point.x, y=point.y)

def getOvalDistance(a: float, b: float, cx: float, cy: float, x: float, y: float) -> float:
    return ((x - cx)**2 / a**2) + ((y - cy)**2 / b**2)

# 尋找大於長度 length 的橢圓，目標長軸為原本長軸 a 的根號多少倍
# 若目標橢圓的長軸為原本的根號 n 倍，代入原本的橢圓公式，會得到 n
def findRootMultiple(a: float, length: float) -> int:
    n = 1
    while a < length:
        n += 1
        a *= math.sqrt(n)
    return n


def gaussian_weight(distance, sigma=1):
    """
    Calculate the weight of a pixel based on its distance from the center using a Gaussian function.

    Parameters:
    - distance: Distance of the pixel from the center.
    - sigma: Standard deviation of the Gaussian function.

    Returns:
    - Weight of the pixel.
    """
    return np.exp(-0.5 * (distance / sigma) ** 2)

# 計算 center 與 point 形成的連線和 X 軸所夾的弧度
def findRadius(center: Point, point: Point) -> float:
    delta_x = point.x - center.x
    delta_y = point.y - center.y
    angle = math.atan2(delta_y, delta_x)
    return angle

# # 橢圓的中心為 center，長軸為 a，短軸為 b，計算和 X 軸的夾角弧度為 radius 時，橢圓邊上的點
# def findOvalPointByRadius(center: Point, a: float, b: float, radius: float) -> Point:
#     pass

def findOvalPointByRadius(center: Point, a: float, b: float, radius: float) -> Point:
    """
    找到直線與橢圓的交點，並選擇正確的點。
    
    參數：
        center: 橢圓的圓心座標 (cx, cy)
        a: 橢圓的長半軸
        b: 橢圓的短半軸
        radius: 直線與 x 軸正方向的夾角，以弧度表示
    
    回傳：
        (x, y) 座標
    """
    # 直線的斜率
    m = math.tan(radius)
    
    # 直線方程式為 y = mx + c
    # 找到 c 的值
    c = center.y - m * center.x
    
    # 求解直線和橢圓的交點
    # 橢圓方程式為 (x-cx)^2/a^2 + (y-cy)^2/b^2 = 1
    A = (a**2) * (m**2) + (b**2)
    B = 2 * a**2 * m * c - 2 * a**2 * center.x * m
    C = a**2 * (c**2 - 2 * c * center.x + center.x**2) - a**2 * b**2
    discriminant = B**2 - 4 * A * C
    
    # 檢查是否有實根
    if discriminant < 0:
        raise ValueError(f"直線與橢圓沒有交點, center: {center}")
    
    # 找到兩個交點的 x 座標
    x1 = (-B + math.sqrt(discriminant)) / (2 * A)
    x2 = (-B - math.sqrt(discriminant)) / (2 * A)
    
    # 根據直線的方向選擇適當的交點
    # 如果直線的斜率為正，則選擇較大的 x 值，否則選擇較小的 x 值
    if m > 0:
        x = max(x1, x2)
    else:
        x = min(x1, x2)
    
    # 使用直線方程式找到對應的 y 值
    y = m * x + c
    
    return Point(x, y)


def findOvalIntersection(center: Point, a: float, b: float, radius: float) -> Point:
    h, k = center.x, center.y
    
    # 處理斜率無窮大的情況
    if math.isclose(radius % math.pi, math.pi / 2, rel_tol=1e-9):
        # 直線方程為 x = h，解橢圓方程 ((h - h)^2) / a^2 + ((y - k)^2) / b^2 = 1
        y1 = k + b
        y2 = k - b
        if radius >= 0 and radius < math.pi:
            return Point(h, y1)
        else:
            return Point(h, y2)
    
    tan_radius = math.tan(radius)
    
    # 橢圓方程：((x - h)^2) / a^2 + ((y - k)^2) / b^2 = 1
    # 直線方程：y - k = tan(radius) * (x - h)
    # 替換 y 的值：
    # ((x - h)^2) / a^2 + (tan(radius) * (x - h))^2 / b^2 = 1
    
    A = 1 / a**2 + (tan_radius**2) / b**2
    B = -2 * h / a**2 - 2 * (tan_radius**2) * h / b**2
    C = (h**2) / a**2 + (h**2) * (tan_radius**2) / b**2 - 1
    
    # 解二次方程 Ax^2 + Bx + C = 0
    discriminant = B**2 - 4 * A * C
    
    if discriminant < 0:
        raise ValueError("No intersection points found.")
    
    x1 = (-B + math.sqrt(discriminant)) / (2 * A)
    x2 = (-B - math.sqrt(discriminant)) / (2 * A)
    
    y1 = k + tan_radius * (x1 - h)
    y2 = k + tan_radius * (x2 - h)
    
    # 選擇其中一個交點，我們選擇與 radius 同方向的交點
    if radius >= 0 and radius < math.pi:
        return Point(x1, y1)
    else:
        return Point(x2, y2)