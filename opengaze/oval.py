import math
from coordinate import Point
from utils import findRadius, radiansToSlope


class Oval:
    def __init__(self, center: Point, a: float, b: float):
        self.center = center
        self.a = a
        self.b = b
        self.x2 = center.x**2
        self.a2 = a**2
        self.b2 = b**2

    def __repr__(self):
        return f"Oval(center: {self.center}, a: {self.a}, b: {self.b})"
    
    __str__ = __repr__

    def getElliptic(self, point: Point) -> float:
        return ((point.x - self.center.x)**2 / self.a**2) + ((point.y - self.center.y)**2 / self.b**2)
    
    def getByRadius(self, radius: float) -> Point:
        tan_radius = radiansToSlope(radius=radius)
  
        # 處理斜率無窮大的情況
        if tan_radius == float('inf'):
            return Point(self.center.x, self.center.y + self.b)

        elif tan_radius == float('-inf'):
            return Point(self.center.x, self.center.y - self.b)

        tan_radius2 = tan_radius**2
        # 橢圓方程：((x - h)^2) / a^2 + ((y - k)^2) / b^2 = 1
        # 直線方程：y - k = tan(radius) * (x - h)
        # 替換 y 的值：
        # ((x - h)^2) / a^2 + (tan(radius) * (x - h))^2 / b^2 = 1
        tan_radius2_divide_by_b2 = tan_radius2 / self.b2
        A = 1 / self.a2 + tan_radius2_divide_by_b2
        B = -2 * self.center.x / self.a2 - 2 * tan_radius2 * self.center.x / self.b2
        C = self.x2 / self.a2 + self.x2 * tan_radius2_divide_by_b2 - 1
        
        # 解二次方程 Ax^2 + Bx + C = 0
        discriminant = B**2 - 4 * A * C
        
        if discriminant < 0:
            raise ValueError("No intersection points found.")
        
        sqrt_discriminant = math.sqrt(discriminant)
        _2A = 2 * A
        
        x1 = (-B + sqrt_discriminant) / _2A
        x2 = (-B - sqrt_discriminant) / _2A
        
        y1 = self.center.y + tan_radius * (x1 - self.center.x)
        y2 = self.center.y + tan_radius * (x2 - self.center.x)
        
        # 選擇其中一個交點，我們選擇與 radius 同方向的交點
        if (0 <= radius and radius <= math.pi / 2) or (1.5 * math.pi <= radius and radius <= 2 * math.pi):
            return Point(x1, y1)
        else:
            return Point(x2, y2)

def getOvalPointDistance(a: float, b: float, center: Point, point: Point) -> float:
    return getOvalDistance(a=a, b=b, cx=center.x, cy=center.y, x=point.x, y=point.y)


def getOvalDistance(a: float, b: float, cx: float, cy: float, x: float, y: float) -> float:
    return ((x - cx)**2 / a**2) + ((y - cy)**2 / b**2)


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


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    center = Point(0, 0)
    oval = Oval(center=center, a=5, b=3)
    print(f"Elliptic(5, 0): {oval.getElliptic(Point(5, 0))}")
    print(f"Elliptic(5, 0): {oval.getElliptic(Point(10, 0))}")

    
    theta = np.linspace(0, 2 * np.pi, 12)
    radiuses = [math.tan(radius) for radius in theta]
    print(f"radiuses: {radiuses}")

    num_points = 100
    theta = np.linspace(0, np.pi / 2, num_points)
    x = []
    y = []
    for radius in theta:
        point = oval.getByRadius(radius)

        r = findRadius(center=center, point=point)
        if math.fabs(math.fabs(radius) - math.fabs(r)) > 1e-7:
            print(f"radius: {radius}, r: {r}")
            
        x.append(point.x)
        y.append(point.y)
    plt.scatter(x, y, c='r')

    
    theta = np.linspace(np.pi / 2, np.pi, num_points)
    x = []
    y = []
    for radius in theta:
        point = oval.getByRadius(radius)
        x.append(point.x)
        y.append(point.y)
    plt.scatter(x, y, c='g')

    
    theta = np.linspace(np.pi, 1.5 * np.pi, num_points)
    x = []
    y = []
    for radius in theta:
        point = oval.getByRadius(radius)
        x.append(point.x)
        y.append(point.y)
    plt.scatter(x, y, c='blue')

    
    theta = np.linspace(1.5 * np.pi, 2 * np.pi, num_points)
    x = []
    y = []
    for radius in theta:
        point = oval.getByRadius(radius)
        x.append(point.x)
        y.append(point.y)
    plt.scatter(x, y, c='black')

    # 添加图形标题和坐标轴标签
    plt.title('Scatter Plot of sin(theta)')
    plt.xlabel('Theta (radians)')
    plt.ylabel('sin(theta)')

    # 显示图形
    plt.show()

    