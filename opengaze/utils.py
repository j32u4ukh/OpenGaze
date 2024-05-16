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