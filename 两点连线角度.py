import math


def calculate_rotation_angle(point1, point2):
    # 解构点坐标
    x1, y1 = point1
    x2, y2 = point2

    # 计算dx和dy（图像坐标系中的差值）
    dx = x2 - x1
    dy = y2 - y1  # 注意：OpenCV的y轴向下，因此直接使用y2 - y1

    # 计算弧度
    angle_rad = math.atan2(dy, dx)

    # 转换为度数
    angle_deg = math.degrees(angle_rad)

    # 将角度调整到0~360度范围
    if angle_deg < 0:
        angle_deg += 360.0

    return angle_deg


# 示例使用
if __name__ == "__main__":
    # 示例点坐标（格式：(x, y)）
    pointA = (0, 0)
    pointB = (100, 100)  # 右下方，期望角度：45度

    angle = calculate_rotation_angle(pointA, pointB)
    print(f"旋转角度为: {angle:.2f} 度")