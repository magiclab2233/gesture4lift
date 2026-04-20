"""
无接触智慧控制器 - 全局配置
"""

# ==================== 摄像头配置 ====================
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# ==================== MediaPipe Hands 配置 ====================
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# ==================== 手势识别配置 ====================
# 手指伸出判定：指尖到手腕距离 / 指根到手腕距离 的阈值
FINGER_EXTEND_RATIO = 1.15

# 手掌朝向判定：向量叉积阈值（>0 表示手心朝向摄像头）
PALM_FACING_THRESHOLD = 0.02

# 捏合点击判定：食指+拇指捏合的相对阈值（手掌尺寸的 30% 以内）
# 已在代码中硬编码为 palm_size * 0.30，此处保留作为参考
PINCH_THRESHOLD = 0.05

# 手势防抖：连续多少帧识别到同一手势才确认
GESTURE_BUFFER_SIZE = 8

# 点击冷却帧数，防止重复触发
CLICK_COOLDOWN_FRAMES = 15

# ==================== UI 配置 ====================
# Windows 下常见中文字体文件名映射
UI_FONT = "微软雅黑"  # 可选：微软雅黑 / 黑体 / 宋体
UI_FONT_SIZE = 30
UI_SCALE = 1.0
