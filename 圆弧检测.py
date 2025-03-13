import cv2
import numpy as np
from show import show
def detect_circle(img_path):
    img = cv2.imread(img_path)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # show(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # show(gray)
    # 对滤波结果做边缘检测获取目标
    edged = cv2.Canny(gray, 150, 200)
    # show(edged)
    # 使用膨胀和腐蚀操作进行闭合对象边缘之间的间隙
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)
    edged = cv2.dilate(edged, kernel, iterations=1)
    #show(edged)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 霍夫圆变换检测圆弧
    circles = cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=50, param2=40, minRadius=50, maxRadius=0)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        image1 = img.copy()
        max_radius = 0
        largest_circle = None
        # 筛选有效圆形（根据位置约束）
        valid_circles = []
        h, w = img.shape[:2]
        for (x, y, r) in circles:
            if r > max_radius:
                max_radius = r
                largest_circle = (x, y, r)
        if largest_circle:
            x, y, r = largest_circle
            tip_point = (x, y)
            cv2.circle(image1, tip_point, 10, (0, 0, 255), -1)
            cv2.putText(image1, f'Tip1: {tip_point}', (tip_point[0] + 15, tip_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        show(image1)
        return x,y
if __name__ == "__main__":
    detect_circle('image/base.jpg')