import cv2
import numpy as np
from show import show
from 圆弧检测 import detect_circle
from 测剪刀尖 import detect_scissors_tip
from 两点连线角度 import calculate_rotation_angle
def detect_scissor_pose(img):
    #基点1（中心点）
    x1,y1 = detect_circle(img)
    #基点2
    x2,y2 = detect_scissors_tip(img)
    angle = calculate_rotation_angle((x1,y1),(x2,y2))
    return (x1, x2), angle
# 计算基准位置和角度
base_position, base_angle = detect_scissor_pose('result/result.jpg')
print(f"基准位置：{base_position}")
print(f"基准角度：{base_angle}°\n")


# 处理后续4张图像
for i in range(1, 3):


    # 检测当前位置和角度
    curr_position, curr_angle = detect_scissor_pose(f"result/result{i}.jpg")

    if curr_position is None:
        print(f"图像 {i} 未检测到剪刀")
        continue

    # 计算位置偏移（像素坐标）
    dx = curr_position[0] - base_position[0]
    dy = curr_position[1] - base_position[1]

    # 计算角度偏移
    dz = curr_angle - base_angle

    # 输出结果（可根据需要转换为实际物理单位）
    print(f"图像 {i} 位姿:")
    print(f"位置偏移: ({dx, dy})")
    print(f"旋转角度: {dz}°\n")


