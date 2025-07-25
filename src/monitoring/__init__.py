"""
系统监控模块，负责采集系统资源使用情况并记录日志

包括CPU使用率、内存占用、网络IO等关键指标的监控
"""

from .monitor import SystemMonitor

__all__ = ["SystemMonitor"]
__version__ = "1.0.0"
