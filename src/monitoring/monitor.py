import os
import time
import threading
import socket
from datetime import datetime
from typing import Optional, Dict, Any
import psutil

from src.system.base_component import BaseComponent
from src.config.config_manager import ConfigManager
from src.utils.logger import get_logger

class SystemMonitor(BaseComponent):
    """系统监控组件"""
    
    # 添加类级别的停止标志
    _force_stopped = False
    # 添加全局停止标志
    _global_stopped = False
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ConfigManager()
        self.logger = get_logger(__name__)
        
        # 监控间隔（秒）
        self.interval = self.config.get("monitoring.interval", 10)
        
        # 日志配置
        self.log_path = self.config.get("monitoring.log_path", "logs/monitor.log")
        self.hostname = socket.gethostname()
        
        # 监控线程和停止事件
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # 网络监控时间戳和初始值
        self._last_net_time = time.time()
        counters = psutil.net_io_counters()
        self._last_net_bytes_sent = counters.bytes_sent
        self._last_net_bytes_recv = counters.bytes_recv
        
        # 最近的监控数据
        self._latest_metrics: Optional[Dict[str, Any]] = None
        
        self.logger.info(f"系统监控组件初始化完成，监控间隔: {self.interval}秒")

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
        disk_io_stats = {
            "read_count": 0,
            "write_count": 0,
            "read_bytes": 0,
            "write_bytes": 0
        }
        
        if disk_io is not None:
            disk_io_stats.update({
                "read_count": disk_io.read_count,
                "write_count": disk_io.write_count,
                "read_bytes": disk_io.read_bytes,
                "write_bytes": disk_io.write_bytes
            })
            
        return {
            "usage": disk_usage,
            "io": disk_io_stats
        }

    def _monitor_loop(self):
        """监控主循环"""
        # 检查是否已被强制停止
        if SystemMonitor._force_stopped or SystemMonitor._global_stopped:
            self.logger.debug("监控循环检测到系统已被强制停止或全局停止，不启动监控循环")
            return
            
        self.logger.debug("监控循环开始")
        iteration = 0
        try:
            while self._is_running and not self._stop_event.is_set() and not SystemMonitor._force_stopped and not SystemMonitor._global_stopped:
                iteration += 1
                self.logger.debug(f"监控循环第 {iteration} 次迭代开始")
                
                # 检查停止标志文件（高频率检查）
                stop_flag_file = ".system_stop_flag"
                for _ in range(10):  # 在每次主要循环中检查多次
                    if os.path.exists(stop_flag_file):
                        self.logger.info("检测到系统停止标志文件，正在停止监控")
                        SystemMonitor._global_stopped = True
                        # 尝试删除停止标志文件
                        try:
                            os.remove(stop_flag_file)
                        except:
                            pass
                        # 立即退出循环
                        return
                    # 检查事件是否被设置，如果被设置则立即退出
                    if self._stop_event.is_set() or not self._is_running or SystemMonitor._force_stopped or SystemMonitor._global_stopped:
                        self.logger.debug("在检查停止标志文件期间检测到停止条件")
                        return
                    time.sleep(0.1)  # 每100ms检查一次
                
                # 在循环开始时就检查停止条件
                if self._stop_event.is_set() or not self._is_running or SystemMonitor._force_stopped or SystemMonitor._global_stopped:
                    self.logger.debug("循环开始时检测到停止事件或组件未运行或已被强制停止或全局停止，退出监控循环")
                    break
                
                try:
                    # 检查停止事件，确保能及时退出
                    if self._stop_event.is_set() or not self._is_running or SystemMonitor._force_stopped or SystemMonitor._global_stopped:
                        self.logger.debug("检测到停止事件或组件未运行或已被强制停止或全局停止，退出监控循环")
                        break
                        
                    self.logger.debug("开始采集系统指标")
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
                    
                    # 检查是否应该停止（在记录日志前进行检查）
                    if self._stop_event.is_set() or not self._is_running or SystemMonitor._force_stopped or SystemMonitor._global_stopped:
                        self.logger.debug("准备记录日志前检测到停止条件，退出监控循环")
                        break
                    
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
                    
                    self.logger.debug(f"监控循环耗时: {elapsed:.2f}s, 休眠时间: {sleep_time:.2f}s")
                    
                    # 使用带超时的等待，以便能及时响应停止信号
                    if sleep_time > 0:
                        # 分段等待，每0.1秒检查一次停止信号
                        waited = 0.0
                        while waited < sleep_time:
                            # 每次循环都检查所有停止条件
                            if self._stop_event.is_set() or not self._is_running or SystemMonitor._force_stopped or SystemMonitor._global_stopped:
                                self.logger.debug("在休眠期间收到停止信号或已被强制停止或全局停止，退出监控循环")
                                return
                                                
                            wait_chunk = min(0.1, sleep_time - waited)
                            self.logger.debug(f"等待 {wait_chunk:.2f} 秒")
                            # 使用wait(timeout)期间持续检查停止条件
                            if self._stop_event.wait(timeout=wait_chunk) or SystemMonitor._force_stopped or SystemMonitor._global_stopped:
                                self.logger.debug("在休眠期间收到停止信号或已被强制停止或全局停止，退出监控循环")
                                return
                                                
                            waited += wait_chunk
                    
                    # 在每次迭代结束时都检查停止事件和运行状态
                    if self._stop_event.is_set() or not self._is_running or SystemMonitor._force_stopped or SystemMonitor._global_stopped:
                        self.logger.debug("循环结束时检测到停止事件或组件未运行或已被强制停止或全局停止，退出监控循环")
                        break
                        
                    # 添加显式的yield点，提高响应性
                    if not self._is_running or self._stop_event.is_set() or SystemMonitor._force_stopped or SystemMonitor._global_stopped:
                        self.logger.debug("迭代结束时检测到停止状态，提前退出")
                        break
                        
                    self.logger.debug(f"监控循环第 {iteration} 次迭代结束")
                        
                except Exception as e:
                    self.logger.error(f"监控循环出错: {str(e)}", exc_info=True)
                    # 异常处理期间增加停止信号检查频率
                    if self._stop_event.is_set() or not self._is_running or SystemMonitor._force_stopped or SystemMonitor._global_stopped:
                        self.logger.debug("出错后检测到停止事件或组件未运行或已被强制停止或全局停止，退出监控循环")
                        break
                        
                    # 增加异常处理期间的停止信号检查频率
                    check_interval = 0.001  # 1毫秒
                    end_time = time.time() + 0.01  # 总共等待10毫秒
                    
                    while time.time() < end_time:
                        if self._stop_event.is_set() or not self._is_running or SystemMonitor._force_stopped or SystemMonitor._global_stopped:
                            self.logger.debug("出错后等待期间收到停止信号或已被强制停止或全局停止，退出监控循环")
                            return
                        time.sleep(check_interval)
            
            self.logger.debug("监控循环结束")
            
        except Exception as e:
            self.logger.error(f"监控循环发生未捕获异常: {str(e)}", exc_info=True)
            # 在异常情况下也检查停止条件
            if self._stop_event.is_set() or not self._is_running or SystemMonitor._force_stopped or SystemMonitor._global_stopped:
                self.logger.debug("异常处理期间检测到停止条件")
        finally:
            self.logger.debug("监控循环线程已退出")
            # 确保在循环结束时清除所有运行状态
            self._is_running = False
            self._stop_event.set()  # 确保事件处于已设置状态

    def get_latest_metrics(self):
        """获取最近一次的监控指标"""
        return self._latest_metrics.copy() if self._latest_metrics else {}

    def start(self):
        """启动系统监控"""
        if self._is_running:
            self.logger.warning("系统监控已在运行中")
            return
            
        if SystemMonitor._force_stopped or SystemMonitor._global_stopped:
            self.logger.warning("系统监控已被强制停止或全局停止，无法重新启动")
            return
            
        super().start()
        self._stop_event.clear()  # 清除停止事件，确保线程能正常运行
        
        # 启动监控线程
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        
        self.logger.info(f"系统监控已启动，监控间隔: {self.interval}秒")

    def stop(self):
        """停止系统监控"""
        self.logger.debug("SystemMonitor.stop() 方法被调用")
        if not self._is_running:
            self.logger.debug("系统监控未运行，无需停止")
            return
            
        self.logger.debug("正在停止系统监控")
        # 设置强制停止标志
        SystemMonitor._force_stopped = True
        SystemMonitor._global_stopped = True
        
        # 先设置_is_running为False，确保循环能检测到
        self._is_running = False
        
        # 设置停止事件
        self._stop_event.set()  
        self.logger.debug("已设置停止事件")
        
        # 等待监控线程结束，但设置一个合理的超时时间
        if self._monitor_thread and self._monitor_thread.is_alive():
            import threading
            # 检查是否在监控线程内部调用
            if threading.current_thread() is self._monitor_thread:
                self.logger.debug("在监控线程内部调用停止，无需等待自身")
            else:
                self.logger.debug(f"等待监控线程结束，线程ID: {self._monitor_thread.ident}")
                
                # 设置合理的最大等待时间
                max_wait_time = 10  # 等待时间10秒
                
                # 等待线程结束
                self._monitor_thread.join(timeout=max_wait_time)
                
                if self._monitor_thread.is_alive():
                    self.logger.warning("监控组件线程未能正常终止")
                    self.logger.debug("监控线程可能仍在运行，但已超出最大等待时间")
                else:
                    self.logger.debug("监控组件线程已正常终止")
        else:
            self.logger.debug("监控线程不存在或已停止")
        
        self.logger.info("系统监控已停止")
        
    @classmethod
    def global_stop(cls):
        """全局停止所有SystemMonitor实例"""
        cls._global_stopped = True

    def get_status(self) -> Dict[str, Any]:
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