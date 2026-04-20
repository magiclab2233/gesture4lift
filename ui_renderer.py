"""
UI 渲染模块：在视频帧上叠加控制面板、状态提示、手势可视化。
使用 PIL 支持中文显示。
"""

from typing import Dict, Optional, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
from elevator_animator import ElevatorAnimator


class UIRenderer:
    """
    负责在 OpenCV 图像上绘制 HUD（抬头显示）界面。
    通过 PIL 绘制文字，完美支持中英文混排。
    """

    # 颜色定义 (BGR)
    COLOR_BG = (30, 30, 30)
    COLOR_PANEL = (50, 50, 50)
    COLOR_TEXT = (255, 255, 255)
    COLOR_ACCENT = (0, 200, 255)
    COLOR_SUCCESS = (0, 255, 100)
    COLOR_WARNING = (0, 100, 255)

    def __init__(self):
        self.scale = config.UI_SCALE
        self._font_path = self._find_chinese_font()

    @staticmethod
    def _find_chinese_font() -> str:
        """查找系统中可用的中文字体，优先匹配 config.UI_FONT 设置。"""
        font_map = {
            "微软雅黑": [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc"],
            "黑体":    [r"C:\Windows\Fonts\simhei.ttf"],
            "宋体":    [r"C:\Windows\Fonts\simsun.ttc"],
        }
        candidates = font_map.get(config.UI_FONT, [])
        # 兜底：遍历所有常见字体
        candidates += [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\msyhbd.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\simsun.ttc",
        ]
        for path in candidates:
            if cv2.os.path.exists(path):
                return path
        return ""

    def _get_font(self, size: int):
        """获取指定大小的字体对象。"""
        if self._font_path:
            return ImageFont.truetype(self._font_path, size)
        return ImageFont.load_default()

    def _put_text(self, img: np.ndarray, text: str, pos: Tuple[int, int],
                  font_size: int, color: Tuple[int, int, int]) -> np.ndarray:
        """
        使用 PIL 在 OpenCV 图像上绘制文字（支持中文）。
        返回绘制后的新图像。
        """
        if not text:
            return img

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        font = self._get_font(font_size)

        # PIL 使用 RGB，需要把 BGR 转成 RGB
        rgb_color = (color[2], color[1], color[0])
        draw.text(pos, text, font=font, fill=rgb_color)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _text_size(self, text: str, font_size: int) -> Tuple[int, int]:
        """计算给定文本在指定字号下的像素宽高。"""
        font = self._get_font(font_size)
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def render(
        self,
        image: np.ndarray,
        result: Optional[Dict],
        controller_status: str,
        event_msg: str,
        animator: ElevatorAnimator,
    ) -> np.ndarray:
        """主渲染入口。"""
        canvas = image.copy()

        # 1. 顶部状态栏
        canvas = self._draw_header(canvas, controller_status)

        # 2. 左侧手势说明
        canvas = self._draw_gesture_guide(canvas)

        # 3. 右侧电梯井道动画
        canvas = animator.draw(canvas)

        # 4. 右侧/底部事件提示
        canvas = self._draw_event_toast(canvas, event_msg)

        # 5. 若识别到手势，绘制动态反馈
        if result:
            canvas = self._draw_hand_feedback(canvas, result)

        # 6. 底部操作提示
        canvas = self._draw_footer(canvas)

        return canvas

    def _draw_header(self, canvas: np.ndarray, status: str) -> np.ndarray:
        h, w = canvas.shape[:2]
        bar_h = int(50 * self.scale)
        cv2.rectangle(canvas, (0, 0), (w, bar_h), self.COLOR_BG, -1)
        cv2.line(canvas, (0, bar_h), (w, bar_h), self.COLOR_ACCENT, 2)

        canvas = self._put_text(canvas, "无接触智慧电梯", (15, int(8 * self.scale)),
                                int(22 * self.scale), self.COLOR_ACCENT)
        canvas = self._put_text(canvas, status, (int(w * 0.4), int(8 * self.scale)),
                                int(22 * self.scale), self.COLOR_TEXT)
        return canvas

    def _draw_gesture_guide(self, canvas: np.ndarray) -> np.ndarray:
        h, w = canvas.shape[:2]
        panel_w = int(200 * self.scale)
        panel_h = int(260 * self.scale)
        x1, y1 = 15, int(70 * self.scale)
        x2, y2 = x1 + panel_w, y1 + panel_h

        # 半透明背景
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.COLOR_PANEL, -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), self.COLOR_ACCENT, 1)

        lines = [
            "手势说明：",
            "  1 → 1 楼",
            "  2 → 2 楼",
            "  3 → 3 楼",
            "  4 → 4 楼",
            "  5 → 5 楼",
            "",
            "触发方式：",
            "  五个指尖并拢确认",
        ]
        y = y1 + int(10 * self.scale)
        for line in lines:
            canvas = self._put_text(canvas, line, (x1 + 10, y),
                                    int(16 * self.scale), self.COLOR_TEXT)
            y += int(22 * self.scale)
        return canvas

    def _draw_event_toast(self, canvas: np.ndarray, event_msg: str) -> np.ndarray:
        h, w = canvas.shape[:2]
        if not event_msg:
            return canvas

        font_size = int(20 * self.scale)
        tw, th = self._text_size(event_msg, font_size)
        pad = 15
        x1 = max(10, w - tw - pad * 2 - 20)
        y1 = max(10, h - th - pad * 2 - 40)
        x2, y2 = w - 10, h - 40

        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.COLOR_SUCCESS, -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), self.COLOR_SUCCESS, 2)

        canvas = self._put_text(canvas, event_msg, (x1 + pad, y1 + pad),
                                font_size, self.COLOR_BG)
        return canvas

    def _draw_hand_feedback(self, canvas: np.ndarray, result: Dict) -> np.ndarray:
        h, w = canvas.shape[:2]
        landmarks = result.get("landmarks")
        gesture = result.get("gesture_number", -1)
        confirmed = result.get("confirmed_gesture", -1)
        together = result.get("together", False)
        palm_facing = result.get("palm_facing", False)

        # 在手部附近绘制大数字
        if landmarks and gesture != -1:
            wrist = landmarks[0]
            index_tip = landmarks[8]
            cx = int((wrist[0] + index_tip[0]) / 2 * w)
            cy = int((wrist[1] + index_tip[1]) / 2 * h)

            color = self.COLOR_ACCENT if confirmed != -1 else self.COLOR_WARNING
            text = str(gesture)
            if not palm_facing:
                text = "X"
                color = (0, 0, 255)

            canvas = self._put_text(canvas, text, (cx - 20, cy - 60),
                                    int(48 * self.scale), color)

        # 绘制触发反馈圆圈
        if together:
            center = (w // 2, h // 2)
            radius = 40
            color = self.COLOR_SUCCESS
            cv2.circle(canvas, center, radius, color, 3)
            label = "并拢触发"
            tw, th = self._text_size(label, int(18 * self.scale))
            canvas = self._put_text(canvas, label,
                                    (center[0] - tw // 2, center[1] - radius - th - 10),
                                    int(18 * self.scale), color)
        return canvas

    def _draw_footer(self, canvas: np.ndarray) -> np.ndarray:
        h, w = canvas.shape[:2]
        bar_h = int(36 * self.scale)
        cv2.rectangle(canvas, (0, h - bar_h), (w, h), self.COLOR_BG, -1)
        canvas = self._put_text(canvas, "按 [Q] 退出",
                                (15, h - bar_h + int(5 * self.scale)),
                                int(16 * self.scale), self.COLOR_TEXT)
        return canvas
