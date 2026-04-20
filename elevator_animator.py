"""
电梯移动动画模块
在 OpenCV 图像上绘制电梯井道与轿厢平滑移动动画。
"""

from typing import Optional, Tuple

import cv2
import numpy as np


class ElevatorAnimator:
    """
    负责计算电梯轿厢的平滑位置，并在视频帧上绘制井道动画。
    """

    # 配色 (BGR)
    COLOR_SHAFT = (80, 80, 80)
    COLOR_SHAFT_BORDER = (160, 160, 160)
    COLOR_FLOOR_LABEL = (255, 255, 255)
    COLOR_FLOOR_ACTIVE = (0, 200, 255)
    COLOR_FLOOR_TARGET = (0, 100, 255)
    COLOR_CAR_BODY = (0, 150, 255)
    COLOR_CAR_BORDER = (0, 200, 255)
    COLOR_CAR_ARRIVED = (0, 255, 100)
    COLOR_DOOR = (40, 40, 40)

    def __init__(self, fps: int = 30):
        self.fps = fps
        self.animating: bool = False
        self.from_floor: float = 1.0
        self.to_floor: int = 1
        self.progress: float = 0.0
        self.speed: float = 0.04  # 每帧进度，约 0.8 秒/层
        self.just_arrived: bool = False
        self.arrival_timer: int = 0  # 到达后绿色高亮持续帧数

    def start(self, from_floor: int, to_floor: int):
        """启动从 from_floor 到 to_floor 的动画。"""
        if from_floor == to_floor:
            return
        self.animating = True
        self.from_floor = float(from_floor)
        self.to_floor = to_floor
        self.progress = 0.0
        self.just_arrived = False
        self.arrival_timer = 0

    def update(self) -> bool:
        """
        每帧调用，更新动画进度。
        返回 True 表示本帧刚刚到达目标楼层。
        """
        self.just_arrived = False

        if self.animating:
            self.progress += self.speed
            if self.progress >= 1.0:
                self.progress = 1.0
                self.animating = False
                self.just_arrived = True
                self.arrival_timer = int(self.fps * 1.5)  # 到达后高亮 1.5 秒
                return True
        elif self.arrival_timer > 0:
            self.arrival_timer -= 1

        return False

    def current_display_floor(self) -> float:
        """
        返回当前用于绘制的轿厢楼层位置（浮点数，含平滑插值）。
        """
        if not self.animating and self.arrival_timer <= 0:
            return float(self.to_floor)
        # ease-in-out smoothstep
        t = self.progress
        t = t * t * (3.0 - 2.0 * t)
        return self.from_floor + (self.to_floor - self.from_floor) * t

    def is_moving(self) -> bool:
        return self.animating

    def draw(self, canvas: np.ndarray) -> np.ndarray:
        """
        在 canvas 上绘制电梯井道、楼层标记与轿厢。
        绘制区域位于画面右侧。
        """
        h, w = canvas.shape[:2]
        # 井道区域参数
        shaft_x = int(w * 0.82)
        shaft_w = int(w * 0.12)
        shaft_top = int(h * 0.15)
        shaft_bottom = int(h * 0.82)

        # 绘制井道外框
        cv2.rectangle(
            canvas,
            (shaft_x, shaft_top),
            (shaft_x + shaft_w, shaft_bottom),
            self.COLOR_SHAFT,
            -1,
        )
        cv2.rectangle(
            canvas,
            (shaft_x, shaft_top),
            (shaft_x + shaft_w, shaft_bottom),
            self.COLOR_SHAFT_BORDER,
            2,
        )

        # 5 个楼层均匀分布（1 楼在最下，5 楼在最上）
        total_floors = 5
        floor_positions = []  # 每个楼层的 y 中心坐标
        usable_h = shaft_bottom - shaft_top
        for i in range(total_floors):
            # 1 楼在底部 (i=0)，5 楼在顶部 (i=4)
            ratio = i / (total_floors - 1)
            fy = int(shaft_bottom - ratio * usable_h)
            floor_positions.append(fy)

        # 绘制楼层标记线与数字
        display_floor = self.current_display_floor()
        rounded_floor = int(round(display_floor))
        for idx, fy in enumerate(floor_positions):
            floor_num = idx + 1
            is_current = (floor_num == rounded_floor)
            is_target = self.animating and (floor_num == self.to_floor)

            # 颜色与粗细：当前楼层和目标楼层高亮
            if is_current or is_target:
                line_color = self.COLOR_FLOOR_ACTIVE
                text_color = self.COLOR_FLOOR_ACTIVE
                line_thickness = 2
                font_thickness = 2
            else:
                line_color = self.COLOR_FLOOR_LABEL
                text_color = self.COLOR_FLOOR_LABEL
                line_thickness = 1
                font_thickness = 1

            # 横线（贯穿井道两侧）
            cv2.line(
                canvas,
                (shaft_x - 15, fy),
                (shaft_x + shaft_w + 15, fy),
                line_color,
                line_thickness,
            )

            # 当前/目标楼层在左侧加小圆点指示
            if is_current or is_target:
                cv2.circle(canvas, (shaft_x - 25, fy), 4, self.COLOR_FLOOR_ACTIVE, -1)

            # 楼层数字（井道左侧，字号加大）
            label = f"{floor_num}F"
            cv2.putText(
                canvas,
                label,
                (shaft_x - 55, fy + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                text_color,
                font_thickness,
            )

        # 计算轿厢位置
        # 将楼层号映射到 y 坐标（1->bottom, 5->top）
        t = (display_floor - 1) / (total_floors - 1)
        car_cy = int(shaft_bottom - t * usable_h)
        car_h = int(usable_h / (total_floors - 1) * 0.7)  # 轿厢高度为层间距的 70%
        car_w = int(shaft_w * 0.75)
        car_x = shaft_x + (shaft_w - car_w) // 2
        car_y1 = car_cy - car_h // 2
        car_y2 = car_cy + car_h // 2

        # 到达后高亮绿色，移动中为橙色
        if self.arrival_timer > 0 and not self.animating:
            car_color = self.COLOR_CAR_ARRIVED
            border_color = self.COLOR_CAR_ARRIVED
        else:
            car_color = self.COLOR_CAR_BODY
            border_color = self.COLOR_CAR_BORDER

        # 绘制轿厢
        cv2.rectangle(
            canvas,
            (car_x, car_y1),
            (car_x + car_w, car_y2),
            car_color,
            -1,
        )
        cv2.rectangle(
            canvas,
            (car_x, car_y1),
            (car_x + car_w, car_y2),
            border_color,
            2,
        )

        # 绘制轿厢内的门缝
        door_x = car_x + car_w // 2
        cv2.line(
            canvas,
            (door_x, car_y1 + 3),
            (door_x, car_y2 - 3),
            self.COLOR_DOOR,
            2,
        )

        # 轿厢内显示当前楼层数字
        floor_text = f"{int(round(display_floor))}"
        text_size = cv2.getTextSize(floor_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = car_x + (car_w - text_size[0]) // 2
        text_y = car_y1 + (car_h + text_size[1]) // 2
        cv2.putText(
            canvas,
            floor_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # 若正在移动，在井道上方绘制方向与目标提示
        if self.animating:
            direction = "UP" if self.to_floor > self.from_floor else "DOWN"
            arrow_color = self.COLOR_CAR_BORDER
            cv2.putText(
                canvas,
                f"{direction} -> {self.to_floor}F",
                (shaft_x - 10, shaft_top - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                arrow_color,
                1,
            )

        return canvas
