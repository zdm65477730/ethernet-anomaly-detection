# 异常检测系统核心包
__version__ = "1.0.0"

# 导出核心模块和类
from src.system.system_manager import SystemManager
from src.capture.packet_capture import PacketCapture
from src.capture.session_tracker import SessionTracker
from src.capture.traffic_analyzer import TrafficAnalyzer
from src.detection.anomaly_detector import AnomalyDetector
from src.training.continuous_trainer import ContinuousTrainer

__all__ = [
    "SystemManager",
    "PacketCapture",
    "SessionTracker",
    "TrafficAnalyzer",
    "AnomalyDetector",
    "ContinuousTrainer",
    "__version__"
]