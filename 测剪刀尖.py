import cv2
import numpy as np
import math
from show import show
from 亮度矫正 import highlight_suppression
def detect_scissors_tip(img_path):
    # 读取图像
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # 对滤波结果做边缘检测获取目标
    edged = cv2.Canny(gray, 50, 100)
    # 使用膨胀和腐蚀操作进行闭合对象边缘之间的间隙
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)
    edged = cv2.dilate(edged, kernel, iterations=1)
    # 查找轮廓
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("未检测到轮廓")
        return

    # 筛选轮廓（面积+长宽比）
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:  # 根据实际尺寸调整
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h
        if 0.3 < aspect_ratio < 3:  # 排除细长干扰物
            valid_contours.append(cnt)

    if not valid_contours:
        print("未找到有效轮廓")
        return

    # 选择最大有效轮廓
    max_contour = max(valid_contours, key=cv2.contourArea)

    # 计算轮廓的矩
    M = cv2.moments(max_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0

    # 计算轮廓凸包
    hull = cv2.convexHull(max_contour, returnPoints=False)
    defects = cv2.convexityDefects(max_contour, hull)

    # 寻找最远凸缺陷点
    max_distance = 0
    tip_point = None

    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            far = tuple(max_contour[f][0])
            distance = np.linalg.norm(np.array(far) - np.array([cx, cy]))

            if distance > max_distance:
                max_distance = distance
                tip_point = far

    # 几何验证
    if tip_point is not None:
        # 创建ROI验证区域
        x, y = tip_point
        try:
            roi = edged[y - 20:y + 20, x - 20:x + 20]
            if np.count_nonzero(roi) < 50:  # 排除孤立点
                tip_point = None
        except IndexError:
            tip_point = None

    # 备用策略：轮廓极值点
    if tip_point is None:
        leftmost = tuple(max_contour[max_contour[:, :, 0].argmin()][0])
        rightmost = tuple(max_contour[max_contour[:, :, 0].argmax()][0])
        topmost = tuple(max_contour[max_contour[:, :, 1].argmin()][0])
        bottommost = tuple(max_contour[max_contour[:, :, 1].argmax()][0])

        # 选择距离图像中心最远的极值点
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        candidates = [leftmost, rightmost, topmost, bottommost]
        tip_point = max(candidates,
                        key=lambda p: np.linalg.norm(np.array(p) - np.array(center)))

    # 标记结果
    cv2.circle(img, tip_point, 10, (0, 0, 255), -1)
    cv2.putText(img, f'Tip2: {tip_point}', (tip_point[0] + 15, tip_point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    show(np.hstack([gray, edged]))
    show(img)
    return tip_point

if __name__ == "__main__":
    # 使用示例
    highlight_suppression('image/base.jpg')
    x,y = detect_scissors_tip('result/result.jpg')
    print(x,y)