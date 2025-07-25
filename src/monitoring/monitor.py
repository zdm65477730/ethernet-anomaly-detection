import os
import time
import threading
import psutil
from datetime import datetime
from src.system.base_component import BaseComponent
from src.utils.logger import get_logger, setup_rotating_logger
from src.config.config_manager import ConfigManager

class SystemMonitor(BaseComponent):
    """系统监控器，定期采集系统资源使用情况并写入日志"""
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ConfigManager()
        
        # 监控配置
        self.interval = self.config.get("monitoring.interval", 10)  # 监控间隔(秒)
        self.log_path = self.config.get("monitoring.log_path", "logs/monitor.log")
        
        # 初始化监控日志
        self.logger = setup_rotating_logger(
            "system_monitor",
            self.log_path,
            max_bytes=10*1024*1024,  # 10MB
            backup_count=5
        )
        
        # 系统信息
        self.hostname = os.uname().nodename if hasattr(os, 'uname') else "unknown"
        
        # 监控线程
        self._monitor_thread = None
        
        # 网络IO基准值（用于计算速率）
        self._net_io_counters = psutil.net_io_counters()
        self._last_net_bytes_sent = self._net_io_counters.bytes_sent
        self._last_net_bytes_recv = self._net_io_counters.bytes_recv
        self._last_net_time = time.time()
        
        # 记录最近的监控数据
        self._latest_metrics = {}

    def _get_cpu_usage(self):
        """获取CPU使用率信息"""
        return {
            "overall": psutil.cpu_percent(interval=0.1),
            "per_core": psutil.cpu_percent(interval=0.1, percpu=True),
            "count": psutil.cpu_count(logical=False),
            "logical_count": psutil.cpu_count(logical=True)
        }

    def _get_memory_usage(self):
        """获取内存使用信息"""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            "total": mem.total,
            "available": mem.available,
            "used": mem.used,
            "used_percent": mem.percent,
            "swap_total": swap.total,
            "swap_used": swap.used,
            "swap_percent": swap.percent
        }

    def _get_network_usage(self):
        """获取网络IO信息（计算速率）"""
        current_time = time.time()
        counters = psutil.net_io_counters()
        
        # 计算字节速率
        time_diff = current_time - self._last_net_time
        if time_diff <= 0:
            time_diff = 0.1  # 避免除零错误
            
        sent_rate = (counters.bytes_sent - self._last_net_bytes_sent) / time_diff
        recv_rate = (counters.bytes_recv - self._last_net_bytes_recv) / time_diff
        
        # 更新基准值
        self._last_net_bytes_sent = counters.bytes_sent
        self._last_net_bytes_recv = counters.bytes_recv
        self._last_net_time = current_time
        
        return {
            "bytes_sent": counters.bytes_sent,
            "bytes_recv": counters.bytes_recv,
            "packets_sent": counters.packets_sent,
            "packets_recv": counters.packets_recv,
            "sent_rate": sent_rate,  # 字节/秒
            "recv_rate": recv_rate,  # 字节/秒
            "errin": counters.errin,
            "errout": counters.errout,
            "dropin": counters.dropin,
            "dropout": counters.dropout
        }

    def _get_disk_usage(self):
        """获取磁盘使用信息"""
        # 只监控根目录和日志目录
        paths = ["/", os.path.dirname(self.log_path)]
        disk_usage = {}
        
        for path in paths:
            try:
                usage = psutil.disk_usage(path)
                disk_usage[path] = {
                    "total": usage.total,
                    "used": usage.used,
                    "free": usage.free,
                    "used_percent": usage.percent
                }
            except PermissionError:
                self.logger.warning(f"没有权限访问路径 {path} 的磁盘信息")
            except Exception as e:
                self.logger.error(f"获取路径 {path} 的磁盘信息失败: {str(e)}")
        
        # 获取磁盘IO统计
        disk_io = psutil.disk_io_counters()
        return {
            "usage": disk_usage,
            "io": {
                "read_count": disk_io.read_count,
                "write_count": disk_io.write_count,
                "read_bytes": disk_io.read_bytes,
                "write_bytes": disk_io.write_bytes
            }
        }

    def _monitor_loop(self):
        """监控主循环"""
        while self._is_running:
            try:
                # 采集系统指标
                start_time = time.time()
                
                metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "hostname": self.hostname,
                    "cpu": self._get_cpu_usage(),
                    "memory": self._get_memory_usage(),
                    "network": self._get_network_usage(),
                    "disk": self._get_disk_usage(),
                    "process": {
                        "pid": os.getpid(),
                        "memory": psutil.Process(os.getpid()).memory_info().rss,
                        "cpu": psutil.Process(os.getpid()).cpu_percent(interval=0.1)
                    }
                }
                
                # 更新最近的监控数据
                self._latest_metrics = metrics
                
                # 记录监控日志（INFO级别记录摘要，DEBUG级别记录完整信息）
                log_msg = (
                    f"系统监控 - CPU: {metrics['cpu']['overall']}% "
                    f"内存: {metrics['memory']['used_percent']}% "
                    f"网络: ↑{metrics['network']['sent_rate']/1024:.1f}KB/s "
                    f"↓{metrics['network']['recv_rate']/1024:.1f}KB/s "
                    f"进程内存: {metrics['process']['memory']/1024/1024:.1f}MB"
                )
                self.logger.info(log_msg)
                self.logger.debug(f"完整监控数据: {metrics}")
                
                # 检查资源使用阈值，超过则发出警告
                if metrics['cpu']['overall'] > self.config.get("monitoring.thresholds.cpu", 80):
                    self.logger.warning(f"CPU使用率过高: {metrics['cpu']['overall']}%")
                if metrics['memory']['used_percent'] > self.config.get("monitoring.thresholds.memory", 85):
                    self.logger.warning(f"内存使用率过高: {metrics['memory']['used_percent']}%")
                
                # 计算循环耗时，调整睡眠时间
                elapsed = time.time() - start_time
                sleep_time = max(0, self.interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"监控循环出错: {str(e)}", exc_info=True)
                time.sleep(1)

    def get_latest_metrics(self):
        """获取最近一次的监控指标"""
        return self._latest_metrics.copy() if self._latest_metrics else {}

    def start(self):
        """启动系统监控"""
        if self._is_running:
            self.logger.warning("系统监控已在运行中")
            return
            
        super().start()
        
        # 启动监控线程
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        
        self.logger.info(f"系统监控已启动，监控间隔: {self.interval}秒")

    def stop(self):
        """停止系统监控"""
        if not self._is_running:
            return
            
        super().stop()
        
        # 停止监控线程
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
            if self._monitor_thread.is_alive():
                self.logger.warning("监控线程未能正常终止")
        
        self.logger.info("系统监控已停止")

    def get_status(self):
        """获取监控组件状态"""
        status = super().get_status()
        status.update({
            "interval": self.interval,
            "log_path": self.log_path,
            "hostname": self.hostname,
            "has_latest_metrics": bool(self._latest_metrics),
            "monitor_thread_alive": self._monitor_thread.is_alive() if self._monitor_thread else False
        })
        return status
