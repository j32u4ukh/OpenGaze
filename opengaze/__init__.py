import cv2

from gaze import OpenGaze

"""
OpenCV 的順序為 (width, height), numpy 呈現的順序為 (height, width, 3)
"""

# 讀取圖片
img = cv2.imread("./data/miko.png")
shape = img.shape
print(f"img.shape: {img.shape}")
src_height = shape[0]
src_width = shape[1]

height = int(src_height/3)
width = int(src_width/3)
radius = 30
n_zone = 3
og = OpenGaze(height=height, width=width, radius=radius, n_zone=n_zone)

basic_img = og.basic(img)
print(f"basic_img.shape: {basic_img.shape}")

# x_float = 0.5
# y_float = 0.5
# src_center = Point(og.src_width * x_float, og.src_height * y_float)
# dst_center = Point(og.dst_width * x_float, og.dst_height * y_float)
# pivot = Point(src_center.x + 5, src_center.y + 10)
# dst_point, weight = og.translateOval(src_center, dst_center, pivot, n_zone, a_list, b_list, boundary_list)
# print(f"dst_point: {dst_point}")
# print(f"weight: {weight}")


gaze_x = 0.5
gaze_y = 0.5
dst = og.gazeOval(gaze_x, gaze_y)
print(f"dst.shape: {dst.shape}")

file_name = f"./data/gazeoval-({height}, {width})-({int(gaze_x * 100)}, {int(gaze_y * 100)}).png"
cv2.imwrite(file_name, dst)
print(f"Save as {file_name}")
# cv2.imshow("PON", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
