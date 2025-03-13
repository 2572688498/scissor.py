import cv2
import numpy as np
import matplotlib.pyplot as plt
from show import show
def corner_detection(image):
    # 读取图像
    img = cv2.imread(image)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # show(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # show(gray)
    # 对滤波结果做边缘检测获取目标
    edged = cv2.Canny(gray, 150, 200)
    # show(edged)
    # 使用膨胀和腐蚀操作进行闭合对象边缘之间的间隙
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=3)
    edged = cv2.dilate(edged, kernel, iterations=2)
    show(edged)
    # 进行Harris角点检测
    dst = cv2.cornerHarris(edged, 2, 3, 0.04)

    # 结果膨胀以标记角点
    dst = cv2.dilate(dst, None)
    # 设置阈值，筛选角点
    img[dst > 0.1 * dst.max()] = [0, 255, 255]

    # 显示结果
    show(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
if __name__ == "__main__":
    corner_detection('image/base.jpg')