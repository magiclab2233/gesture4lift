"""
手势训练数据采集工具
=====================
通过摄像头实时采集手部关键点数据，支持按键标注并保存。

使用方式：
    .\\venv\\Scripts\\python.exe data_collector.py

按键说明：
    [0] ~ [5]   保存当前帧为对应数字标签（0=无手指/其他, 1~5=楼层）
    [T]         保存当前帧为"together"（五指并拢确认动作）
    [N]         保存当前帧为"none"（无有效手势）
    [S]         切换是否同时保存原始图像
    [Q]         退出程序

输出文件：
    dataset/labels.csv      标签与21个关键点坐标
    dataset/images/         原始图像（可选）
"""

import csv
import json
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

import config
from gesture_recognizer import HandGestureRecognizer


def main():
    print("=" * 60)
    print("手势训练数据采集工具")
    print("=" * 60)

    # 创建输出目录
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
    images_dir = os.path.join(dataset_dir, "images")
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    csv_path = os.path.join(dataset_dir, "labels.csv")
    csv_exists = os.path.exists(csv_path)

    # 初始化 CSV 文件
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        header = ["timestamp", "label", "img_path"] + [
            f"lm{i}_{axis}" for i in range(21) for axis in ("x", "y", "z")
        ]
        csv_writer.writerow(header)

    # 统计
    stats = {"total": 0, "last_label": "-"}

    # 保存图像开关
    save_images = False

    # 初始化摄像头
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

    if not cap.isOpened():
        print("错误：无法打开摄像头。")
        sys.exit(1)

    print("摄像头初始化完成。")
    print("按 [0]~[5] 标注数字，[T] 标注并拢，[N] 标注无手势，[S] 切换存图，[Q] 退出")
    print("=" * 60)

    recognizer = HandGestureRecognizer()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("警告：帧捕获失败。")
            continue

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 手势识别
        result, annotated = recognizer.process(image_rgb)
        display = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        h, w = display.shape[:2]

        # 绘制采集信息面板
        panel_y = 10
        lines = [
            f"Save Images: {'ON' if save_images else 'OFF'}  (Press S to toggle)",
            f"Total Saved: {stats['total']}",
            f"Last Label:  {stats['last_label']}",
            "",
            "Keys: [0]~[5]=Digit  [T]=Together  [N]=None  [Q]=Quit",
        ]
        for line in lines:
            cv2.putText(display, line, (10, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            panel_y += 25

        # 如果有识别结果，显示详细信息
        if result:
            gesture = result.get("gesture_number", -1)
            together = result.get("together", False)
            palm = result.get("palm_facing", False)
            confirmed = result.get("confirmed_gesture", -1)

            info_text = f"Gesture: {gesture} | Together: {together} | Palm: {palm} | Confirmed: {confirmed}"
            cv2.putText(display, info_text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

        cv2.imshow("Data Collector", display)

        key = cv2.waitKey(1) & 0xFF
        label = None

        if key == ord("q") or key == ord("Q"):
            break
        elif key == ord("s") or key == ord("S"):
            save_images = not save_images
            print(f"保存图像: {'开启' if save_images else '关闭'}")
        elif key == ord("t") or key == ord("T"):
            label = "together"
        elif key == ord("n") or key == ord("N"):
            label = "none"
        elif ord("0") <= key <= ord("5"):
            label = str(key - ord("0"))

        if label and result and result.get("landmarks"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            img_path = ""

            # 可选：保存原始图像
            if save_images:
                img_filename = f"{label}_{timestamp}.jpg"
                img_path = os.path.join("images", img_filename)
                cv2.imwrite(os.path.join(dataset_dir, img_path), frame)

            # 保存 CSV
            landmarks = result["landmarks"]
            row = [timestamp, label, img_path]
            for lm in landmarks:
                row.extend([lm[0], lm[1], lm[2]])
            csv_writer.writerow(row)
            csv_file.flush()

            stats["total"] += 1
            stats["last_label"] = label
            print(f"已保存 [{label}] - 累计 {stats['total']} 条")

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    recognizer.release()
    csv_file.close()

    print(f"\n采集完成！共保存 {stats['total']} 条数据。")
    print(f"数据文件: {csv_path}")
    if save_images:
        print(f"图像目录: {images_dir}")


if __name__ == "__main__":
    main()
