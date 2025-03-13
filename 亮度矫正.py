import cv2
import numpy as np
from show import show
import matplotlib.pyplot as plt
def analyze_histogram(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray],[0],None,[300],[0,300])
    plt.plot(hist)
    plt.show()
def highlight_suppression(img_path):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 检测高光区域（V通道）
    v = hsv[:, :, 2]
    _, mask = cv2.threshold(v, 200, 255, cv2.THRESH_BINARY)  # 调整阈值

    # 修复高光区域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 使用邻近区域颜色填充
    result = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # 显示结果
    show(np.hstack([img, result]))
    cv2.imwrite('result/result2.jpg', result)

if __name__ == "__main__":
    # 使用示例
    #analyze_histogram("image/base.jpg")
    highlight_suppression("image/image_2.jpg")