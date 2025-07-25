"""
系统协调模块，负责管理和协调所有组件的运行

提供统一的启动、停止和状态查询接口，实现故障检测与自动重启
"""

from .base_component import BaseComponent
from .system_manager import SystemManager

__all__ = ["BaseComponent", "SystemManager"]
__version__ = "1.0.0"
