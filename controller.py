"""
控制器逻辑：电梯模式（集成移动动画）
"""

from typing import Dict, Optional
import time

from elevator_animator import ElevatorAnimator


class GestureController:
    """
    将识别到的手势与点击事件映射为电梯控制指令，并管理楼层移动动画。
    """

    def __init__(self):
        # 状态
        self.current_floor: int = 1
        self.target_floor: Optional[int] = None

        # 事件日志
        self.last_event_time: float = 0.0
        self.last_event_msg: str = "系统就绪，请展示手势"

        # 动画器
        self.animator = ElevatorAnimator()
        self.animator.to_floor = self.current_floor

    def handle_gesture(self, result: Dict) -> str:
        """
        根据识别结果更新状态，返回状态描述字符串。
        """
        if result is None:
            return self.status_text()

        confirmed = result.get("confirmed_gesture", -1)
        trigger = result.get("trigger_click", False)
        palm_facing = result.get("palm_facing", False)

        # 动画期间不响应新指令（防止误触）
        if self.animator.is_moving():
            return self.status_text()

        # 只有手心朝向摄像头时才响应手势
        if not palm_facing:
            return self.status_text()

        # 先更新目标（手势缓冲确认后）
        if confirmed != -1:
            if 1 <= confirmed <= 5:
                self.target_floor = confirmed
                self.last_event_msg = f"选中楼层：{confirmed}"
                self.last_event_time = time.time()

        # 触发点击（确认执行）
        if trigger:
            if self.target_floor is not None:
                self.last_event_msg = f"电梯正在前往 {self.target_floor} 楼..."
                self.last_event_time = time.time()
                self.animator.start(self.current_floor, self.target_floor)
                self.target_floor = None
            else:
                self.last_event_msg = "请先比出楼层数字再点击"
                self.last_event_time = time.time()

        return self.status_text()

    def update_animation(self):
        """
        每帧调用，更新电梯动画状态。
        """
        arrived = self.animator.update()
        if arrived:
            self.current_floor = self.animator.to_floor
            self.last_event_msg = f"电梯已到达 {self.current_floor} 楼"
            self.last_event_time = time.time()

    def status_text(self) -> str:
        """生成当前状态文本。"""
        if self.animator.is_moving():
            to_f = self.animator.to_floor
            return f"正在前往 {to_f} 楼..."

        target = f" | 目标：{self.target_floor} 楼" if self.target_floor else " | 等待选择楼层"
        return f"当前楼层：{self.current_floor} 楼{target}"
