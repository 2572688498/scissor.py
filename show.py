import cv2
def show(img):
    # 显示结果
    img = size_change(img)
    cv2.imshow(" ", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def size_change(orig):
    # 设定新的尺寸
    new_width = int(orig.shape[1] * 0.3)  # 宽度减小
    new_height = int(orig.shape[0] * 0.3)  # 高度减小
    new_size = (new_width, new_height)
    # 使用 INTER_AREA 插值方法缩小图像
    return cv2.resize(orig, new_size, interpolation=cv2.INTER_AREA)