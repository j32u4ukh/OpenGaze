import math


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point(x: {self.x}, y: {self.y})"
    
    __str__ = __repr__

    def translate(self, v: "Vector") -> "Point":
        return Point(x = self.x + v.x, y = self.y + v.y)
    
    def computeDistance(self, point: "Point") -> float:
        return math.sqrt((self.x - point.x)**2 + (self.y - point.y)**2)


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
    
    def norm(v: "Vector") -> "Vector":
        length = v.getLength()
        if length == 0:
            return v
        x = v.x / length
        y = v.y / length
        return Vector(p1=Point(0, 0), p2=Point(x, y))