# 异常检测系统核心包
__version__ = "1.0.0"

# 导出核心模块和类
"""
异常检测系统核心模块
"""
from src.system.system_manager import SystemManager
from src.config.config_manager import ConfigManager

__all__ = ["SystemManager", "ConfigManager"]
