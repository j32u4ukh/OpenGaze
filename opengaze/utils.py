import math
import numpy as np

from coordinate import Point    
       

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


# 計算 center 與 point 形成的連線和 X 軸所夾的弧度
def findRadius(center: Point, point: Point) -> float:
    delta_x = point.x - center.x
    delta_y = point.y - center.y
    angle = math.atan2(delta_y, delta_x)
    return angle