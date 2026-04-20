"""
无接触智慧电梯控制器（含楼层移动动画）
主程序入口

运行方式（使用虚拟环境）：
    .\venv\Scripts\python.exe main.py

操作说明：
    - 举起一只手，手心朝向摄像头
    - 比出数字 1-5 选择目标楼层
    - 食指与中指捏合，或食指快速前伸（Z轴点击）确认
    - 按 [Q] 键退出
"""

import sys

import cv2
import numpy as np

import config
from gesture_recognizer import HandGestureRecognizer
from controller import GestureController
from ui_renderer import UIRenderer


def main():
    print("=" * 50)
    print("无接触智慧电梯控制器")
    print("=" * 50)
    print("正在初始化摄像头...")

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

    if not cap.isOpened():
        print("错误：无法打开摄像头。")
        sys.exit(1)

    print("摄像头初始化完成。")
    print("正在加载 MediaPipe HandLandmarker 模型...")

    recognizer = HandGestureRecognizer()
    controller = GestureController()
    renderer = UIRenderer()

    print("就绪。请举起手，手心朝向摄像头进行操作。")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("警告：帧捕获失败。")
            continue

        # 镜像翻转，符合人类直觉
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 手势识别
        result, annotated = recognizer.process(image_rgb)
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        # 控制器逻辑（处理手势输入）
        status_text = controller.handle_gesture(result)

        # 更新电梯移动动画（每帧必调）
        controller.update_animation()

        # UI 渲染（包含动画绘制）
        display = renderer.render(
            annotated_bgr,
            result,
            status_text,
            controller.last_event_msg,
            controller.animator,
        )

        cv2.imshow("Contactless Gesture Controller", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    recognizer.release()
    print("程序已退出。")


if __name__ == "__main__":
    main()
