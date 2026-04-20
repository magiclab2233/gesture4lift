"""
手势识别核心模块
基于 MediaPipe Tasks HandLandmarker，优化了指尖状态判定、手掌朝向过滤、捏合/Z轴点击检测。
"""

import math
import os
from collections import deque
from typing import Optional, Tuple, Dict, List

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions

import config


class HandGestureRecognizer:
    """
    封装 MediaPipe HandLandmarker 与手势解析逻辑。
    """

    # 21 个 Landmark 的语义索引（与 MediaPipe Hands 一致）
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_DIP, INDEX_FINGER_TIP = 5, 6, 7, 8
    MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_DIP, MIDDLE_FINGER_TIP = 9, 10, 11, 12
    RING_FINGER_MCP, RING_FINGER_PIP, RING_FINGER_DIP, RING_FINGER_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

    # 手指对应的 Landmark 列表 (指尖, 第二关节/近端关节, 指根)
    FINGERS = {
        "thumb":  (THUMB_TIP, THUMB_IP, THUMB_MCP),
        "index":  (INDEX_FINGER_TIP, INDEX_FINGER_PIP, INDEX_FINGER_MCP),
        "middle": (MIDDLE_FINGER_TIP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_MCP),
        "ring":   (RING_FINGER_TIP, RING_FINGER_PIP, RING_FINGER_MCP),
        "pinky":  (PINKY_TIP, PINKY_PIP, PINKY_MCP),
    }

    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MediaPipe model not found: {model_path}\n"
                "Please download it from:\n"
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task"
            )

        # Windows C 扩展在部分环境下对含中文路径的 UTF-8 解析不佳，
        # 尝试加载，失败则复制到临时英文目录再加载。
        import shutil
        import tempfile
        load_path = model_path
        try:
            base_options = BaseOptions(model_asset_path=load_path)
            options = HandLandmarkerOptions(
                base_options=base_options,
                num_hands=config.MAX_NUM_HANDS,
                min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                min_hand_presence_confidence=config.MIN_TRACKING_CONFIDENCE,
                min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            )
            self.detector = HandLandmarker.create_from_options(options)
        except FileNotFoundError:
            temp_dir = tempfile.gettempdir()
            load_path = os.path.join(temp_dir, "hand_landmarker.task")
            shutil.copy2(model_path, load_path)
            base_options = BaseOptions(model_asset_path=load_path)
            options = HandLandmarkerOptions(
                base_options=base_options,
                num_hands=config.MAX_NUM_HANDS,
                min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                min_hand_presence_confidence=config.MIN_TRACKING_CONFIDENCE,
                min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            )
            self.detector = HandLandmarker.create_from_options(options)

        # 手部骨架连接关系（用于 OpenCV 自绘）
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # index
            (0, 9), (9, 10), (10, 11), (11, 12),   # middle
            (0, 13), (13, 14), (14, 15), (15, 16), # ring
            (0, 17), (17, 18), (18, 19), (19, 20), # pinky
            (5, 9), (9, 13), (13, 17),             # palm
        ]

        # 防抖缓冲区
        self._gesture_buffer: deque = deque(maxlen=config.GESTURE_BUFFER_SIZE)

        # 点击冷却计数器
        self._click_cooldown = 0

        # 上一帧确认的五指并拢状态
        self._last_together = False

        # 最近一次经防抖确认的手势（用于并拢时保持已选数字）
        self._last_confirmed_gesture: int = -1

    def process(self, image_rgb: np.ndarray) -> Tuple[Optional[Dict], Optional[np.ndarray]]:
        """
        处理单帧图像，返回识别结果与绘制了关键点的图像。

        Returns:
            result: dict 或 None，包含以下字段：
                - "handedness": str, "Left" / "Right" (Tasks API 当前版本不直接提供，留空)
                - "landmarks": List[Tuple[x, y, z]] (21个，均为归一化坐标)
                - "extended_fingers": List[str]，伸出的手指名称列表
                - "palm_facing": bool，是否手心朝向摄像头
                - "gesture_number": int，0-5 的数字手势（-1 表示无效）
                - "together": bool，是否处于五指并拢状态
                - "confirmed_gesture": int，经防抖后的确认手势（-1 表示未确认）
                - "trigger_click": bool，五指并拢上升沿触发信号
            annotated_image: 绘制了手部骨架的图像
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = self.detector.detect(mp_image)

        annotated_image = image_rgb.copy()

        if not detection_result.hand_landmarks:
            self._gesture_buffer.clear()
            self._last_together = False
            return None, annotated_image

        # 只取第一只手
        hand_landmarks_proto = detection_result.hand_landmarks[0]

        # 提取坐标 (x, y, z)
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks_proto]

        # 绘制骨架（将 proto 列表转回需要的格式）
        self._draw_landmarks(annotated_image, hand_landmarks_proto)

        # 1. 指尖状态判定（优化版：基于相对距离）
        extended_fingers = self._detect_extended_fingers(landmarks)

        # 2. 手掌朝向判定
        palm_facing = self._is_palm_facing_camera(landmarks)

        # 3. 手势数字映射
        gesture_number = self._map_to_number(extended_fingers)

        # 4. 五指并拢检测
        together = self._detect_fingers_together(landmarks)

        # 5. 防抖
        confirmed_gesture = self._debounce_gesture(gesture_number)

        # 6. 点击触发（带冷却）：仅五指并拢上升沿
        trigger_click = False
        if self._click_cooldown > 0:
            self._click_cooldown -= 1
        else:
            if together and not self._last_together:
                trigger_click = True
                self._click_cooldown = config.CLICK_COOLDOWN_FRAMES

        self._last_together = together

        result = {
            "handedness": "Unknown",
            "landmarks": landmarks,
            "extended_fingers": extended_fingers,
            "palm_facing": palm_facing,
            "gesture_number": gesture_number,
            "together": together,
            "confirmed_gesture": confirmed_gesture,
            "trigger_click": trigger_click,
        }
        return result, annotated_image

    # ------------------------------------------------------------------ #
    # 绘制辅助
    # ------------------------------------------------------------------ #
    def _draw_landmarks(self, image: np.ndarray, landmarks_proto: List):
        """
        使用 OpenCV 绘制手部骨架。
        """
        h, w = image.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks_proto]

        for start, end in self.hand_connections:
            cv2.line(image, pts[start], pts[end], (255, 0, 0), 2)
        for pt in pts:
            cv2.circle(image, pt, 3, (0, 255, 0), -1)
        # 食指指尖高亮
        if len(pts) > 8:
            cv2.circle(image, pts[8], 5, (0, 200, 255), -1)

    # ------------------------------------------------------------------ #
    # 内部工具方法
    # ------------------------------------------------------------------ #

    @staticmethod
    def _distance_2d(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def _distance_3d(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

    def _detect_extended_fingers(self, landmarks: List[Tuple[float, float, float]]) -> List[str]:
        """
        判定哪些手指伸出。
        优化逻辑：
        - 其他四指：指尖到手腕距离 > 指根到手腕距离 * ratio，
          且指尖必须比第二关节(PIP)更远离手腕（防止半弯曲误判）。
        - 大拇指：指尖到食指根的距离 vs 拇指根到食指根的距离，阈值 0.75。
        """
        wrist = landmarks[self.WRIST]
        extended = []

        for name, (tip_id, pip_id, mcp_id) in self.FINGERS.items():
            if name == "thumb":
                # 大拇指判定：指尖到食指根的距离 vs 拇指根到食指根的距离
                # 阈值 0.75（基于真实采集数据：标签4最大0.53，标签5最小0.91）
                index_mcp = landmarks[self.INDEX_FINGER_MCP]
                tip_to_index = self._distance_2d(landmarks[tip_id], index_mcp)
                mcp_to_index = self._distance_2d(landmarks[mcp_id], index_mcp)
                if tip_to_index > mcp_to_index * 0.75:
                    extended.append(name)
            else:
                tip_to_wrist = self._distance_2d(landmarks[tip_id], wrist)
                mcp_to_wrist = self._distance_2d(landmarks[mcp_id], wrist)
                pip_to_wrist = self._distance_2d(landmarks[pip_id], wrist)
                # 条件1：指尖比指根更远离手腕（基础比例）
                # 条件2：指尖比第二关节更远离手腕（确保手指充分伸直）
                cond1 = tip_to_wrist > mcp_to_wrist * config.FINGER_EXTEND_RATIO
                cond2 = tip_to_wrist > pip_to_wrist * 1.05
                if cond1 and cond2:
                    extended.append(name)

        return extended

    def _is_palm_facing_camera(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """
        通过向量叉积判断手心是否朝向摄像头（手掌展开状态）。
        取手腕 -> 食指根 和 手腕 -> 小指根 两个向量，叉积绝对值足够大则认为手掌展开。
        """
        wrist = landmarks[self.WRIST]
        index_mcp = landmarks[self.INDEX_FINGER_MCP]
        pinky_mcp = landmarks[self.PINKY_MCP]

        v1 = (index_mcp[0] - wrist[0], index_mcp[1] - wrist[1])
        v2 = (pinky_mcp[0] - wrist[0], pinky_mcp[1] - wrist[1])

        cross_z = v1[0] * v2[1] - v1[1] * v2[0]
        # 放宽阈值以支持并拢动作（真实数据中together最小约0.015）
        return abs(cross_z) > 0.014

    def _map_to_number(self, extended_fingers: List[str]) -> int:
        """
        将伸出的手指映射为 0-5 的数字。
        大拇指是否伸出会影响部分手势，但主要靠 index/middle/ring/pinky 的数量。
        """
        main_count = sum(1 for f in ["index", "middle", "ring", "pinky"] if f in extended_fingers)
        thumb_out = "thumb" in extended_fingers

        if main_count == 0:
            return 0
        if main_count == 1:
            return 1
        if main_count == 2:
            return 2
        if main_count == 3:
            return 3
        if main_count == 4:
            if thumb_out:
                return 5
            return 4
        return -1

    def _detect_fingers_together(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """
        检测"五个指尖并拢"确认动作。
        核心特征：
        1. 拇指尖靠近其他四指指尖（排除数字4/5）
        2. 四指指尖离散度小（排除数字3）
        """
        palm_size = self._distance_2d(
            landmarks[self.WRIST], landmarks[self.MIDDLE_FINGER_MCP]
        )

        # 条件1：拇指尖靠近其他四指指尖
        thumb_tip = landmarks[self.THUMB_TIP]
        min_dist = min(
            self._distance_2d(thumb_tip, landmarks[i])
            for i in [self.INDEX_FINGER_TIP, self.MIDDLE_FINGER_TIP,
                      self.RING_FINGER_TIP, self.PINKY_TIP]
        )
        if min_dist >= palm_size * 0.65:
            return False

        # 条件2：四指指尖离散度小
        tips = [landmarks[i] for i in [
            self.INDEX_FINGER_TIP, self.MIDDLE_FINGER_TIP,
            self.RING_FINGER_TIP, self.PINKY_TIP,
        ]]
        cx = sum(t[0] for t in tips) / 4
        cy = sum(t[1] for t in tips) / 4
        avg_dev = sum(
            math.hypot(t[0] - cx, t[1] - cy) for t in tips
        ) / 4
        if avg_dev >= palm_size * 0.30:
            return False

        return True

    def _debounce_gesture(self, gesture_number: int) -> int:
        """
        手势防抖：连续多帧检测到同一手势才确认。
        若当前帧为 -1（如并拢状态），不污染缓冲区，保持上次确认结果。
        返回确认的手势数字，未确认返回 -1。
        """
        if gesture_number == -1:
            return self._last_confirmed_gesture

        self._gesture_buffer.append(gesture_number)
        if len(self._gesture_buffer) < config.GESTURE_BUFFER_SIZE:
            return -1

        first = self._gesture_buffer[0]
        if all(g == first for g in self._gesture_buffer):
            self._last_confirmed_gesture = first
            return first

        # 缓冲区不一致，说明手势在变化，清空重新累计
        self._gesture_buffer.clear()
        return -1

    def release(self):
        self.detector.close()
