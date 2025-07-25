import abc
import time
from typing import Dict, Any

class BaseComponent(metaclass=abc.ABCMeta):
    """
    所有组件的基类，定义统一的接口规范
    
    提供组件生命周期管理（启动/停止）和状态查询功能
    """
    
    def __init__(self):
        self._is_running = False  # 组件运行状态
        self._start_time = None   # 启动时间戳
        self._stop_time = None    # 停止时间戳
        self._error_count = 0     # 错误计数器
        self._last_error = None   # 最后一次错误信息

    @property
    def is_running(self) -> bool:
        """返回组件是否正在运行"""
        return self._is_running

    def start(self) -> None:
        """启动组件"""
        if self._is_running:
            return
            
        self._is_running = True
        self._start_time = time.time()
        self._stop_time = None

    def stop(self) -> None:
        """停止组件"""
        if not self._is_running:
            return
            
        self._is_running = False
        self._stop_time = time.time()

    def record_error(self, error: Exception) -> None:
        """记录组件错误"""
        self._error_count += 1
        self._last_error = {
            "message": str(error),
            "timestamp": time.time(),
            "type": error.__class__.__name__
        }

    def get_status(self) -> Dict[str, Any]:
        """
        获取组件状态信息
        
        返回:
            包含组件状态的字典
        """
        uptime = None
        if self._is_running and self._start_time:
            uptime = time.time() - self._start_time
            
        return {
            "is_running": self._is_running,
            "start_time": self._start_time,
            "stop_time": self._stop_time,
            "uptime": uptime,
            "error_count": self._error_count,
            "last_error": self._last_error,
            "component_type": self.__class__.__name__
        }

    def __str__(self) -> str:
        status = "运行中" if self._is_running else "已停止"
        return f"{self.__class__.__name__} ({status})"
