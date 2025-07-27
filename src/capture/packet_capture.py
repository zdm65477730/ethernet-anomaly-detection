import pcapy
import dpkt
import time
import threading
import traceback
from typing import Optional, Callable, Any
from src.system.base_component import BaseComponent
from src.utils.logger import get_logger

# 尝试导入pcapy-ng，如果失败则导入pcapy
try:
    import pcapy_ng as pcapy
except ImportError:
    import pcapy

class PacketCapture(BaseComponent):
    """数据包捕获组件，负责从网络接口捕获数据包或从离线文件读取数据包并进行初步处理"""

    def __init__(self, interface: str = "eth0", bpf_filter: Optional[str] = None, 
                 offline_file: Optional[str] = None):
        super().__init__()
        self.logger = get_logger("packet_capture")
        self.interface = interface
        self.bpf_filter = bpf_filter
        self.offline_file = offline_file  # 离线pcap文件路径
        
        # 数据包捕获参数
        self.snaplen = 65535  # 最大捕获长度
        self.promisc = 1      # 混杂模式
        self.timeout = 100    # 超时时间(ms)
        
        # pcapy对象
        self.pcap: Optional[pcapy.PcapObject] = None
        
        # 数据包处理队列
        self.packet_queue = []
        self.max_queue_size = 10000
        
        # 捕获线程
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # 统计信息
        self.stats = {
            "packets_captured": 0,
            "packets_dropped": 0,
            "queue_size": 0
        }
        
        # 工作模式：实时捕获或离线文件处理
        self.mode = "offline" if offline_file else "live"
    
    def set_interface(self, interface: str) -> None:
        """设置网络接口"""
        self.interface = interface
        self.mode = "live"
        self.offline_file = None
        
    def set_filter(self, bpf_filter: str) -> None:
        """设置BPF过滤规则"""
        self.bpf_filter = bpf_filter
        if self.pcap and self.bpf_filter:
            try:
                self.pcap.setfilter(self.bpf_filter)
            except Exception as e:
                self.logger.error(f"设置BPF过滤规则失败: {e}")
    
    def set_offline_file(self, file_path: str) -> None:
        """设置离线pcap文件路径"""
        self.offline_file = file_path
        self.mode = "offline"
        self.interface = None
    
    def _capture_loop(self) -> None:
        """数据包捕获循环"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while not self._stop_event.is_set():
            try:
                if self.mode == "live":
                    # 实时捕获模式
                    self.pcap = pcapy.open_live(
                        self.interface,
                        self.snaplen,
                        self.promisc,
                        self.timeout
                    )
                elif self.mode == "offline" and self.offline_file:
                    # 离线文件处理模式
                    self.pcap = pcapy.open_offline(self.offline_file)
                else:
                    self.logger.error("未指定网络接口或离线文件")
                    break
                
                # 设置BPF过滤规则
                if self.bpf_filter:
                    self.pcap.setfilter(self.bpf_filter)
                
                self.logger.info(f"成功打开{'网络接口' if self.mode == 'live' else '离线文件'}: {self.interface or self.offline_file}")
                consecutive_errors = 0  # 重置错误计数
                
                # 开始捕获数据包
                while not self._stop_event.is_set():
                    try:
                        # 读取数据包
                        header, packet = self.pcap.next()
                        if header and packet:
                            # 添加到队列
                            if len(self.packet_queue) < self.max_queue_size:
                                self.packet_queue.append((header, packet))
                                self.stats["packets_captured"] += 1
                            else:
                                self.stats["packets_dropped"] += 1
                                
                            # 更新队列大小统计
                            self.stats["queue_size"] = len(self.packet_queue)
                            
                            # 离线文件处理时增加小延迟以避免CPU占用过高
                            if self.mode == "offline":
                                time.sleep(0.001)  # 1ms延迟
                        elif self.mode == "offline":
                            # 离线文件读取完成
                            self.logger.info("离线文件读取完成")
                            self._stop_event.set()
                            break
                            
                    except Exception as e:
                        self.logger.error(f"读取数据包时出错: {e}")
                        break
                        
            except pcapy.PcapError as e:
                consecutive_errors += 1
                error_msg = str(e)
                
                # 只记录前几次错误，避免日志刷屏
                if consecutive_errors <= 3:
                    self.logger.error(f"抓包循环出错: {self.interface or self.offline_file}: {error_msg}", exc_info=True)
                
                # 特别处理权限问题
                if "Operation not permitted" in error_msg or "permission" in error_msg.lower():
                    self.logger.warning(
                        f"没有权限捕获 {self.interface} 接口的数据包。"
                        "请使用 sudo 运行程序，或将当前用户添加到 pcap/wireshark 用户组。"
                    )
                    # 权限问题通常无法自动恢复，所以停止尝试
                    break
                
                # 如果连续错误太多，停止尝试
                if consecutive_errors >= max_consecutive_errors:
                    self.logger.error(
                        f"连续 {max_consecutive_errors} 次尝试打开接口失败，停止尝试"
                    )
                    break
                    
                # 等待一段时间再重试
                time.sleep(2)
                
            except Exception as e:
                consecutive_errors += 1
                self.logger.error(f"抓包循环出现未预期错误: {e}", exc_info=True)
                
                # 如果连续错误太多，停止尝试
                if consecutive_errors >= max_consecutive_errors:
                    self.logger.error(
                        f"连续 {max_consecutive_errors} 次尝试打开接口失败，停止尝试"
                    )
                    break
                    
                time.sleep(2)
            
            finally:
                # 清理pcap对象
                if self.pcap:
                    try:
                        self.pcap.close()
                    except:
                        pass
                    self.pcap = None
        
        self.logger.info("抓包循环已停止")
    
    def start(self) -> None:
        """启动数据包捕获"""
        if self.is_running:
            self.logger.warning("抓包组件已在运行中")
            return
            
        super().start()
        
        # 启动捕获线程
        self._stop_event.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="PacketCaptureThread"
        )
        self._capture_thread.start()
        
        self.logger.info(f"抓包组件已启动 (模式: {self.mode})")
    
    def stop(self) -> None:
        """停止数据包捕获"""
        if not self.is_running:
            return
            
        # 停止捕获线程
        self._stop_event.set()
        
        # 等待线程结束
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=5)
            if self._capture_thread.is_alive():
                self.logger.warning("抓包线程未能正常终止")
            
        # 清理资源
        if self.pcap:
            try:
                self.pcap.close()
            except Exception as e:
                self.logger.error(f"关闭pcap对象时出错: {e}")
            self.pcap = None
            
        super().stop()
        self.logger.info("抓包组件已停止")
    
    def get_next_packet(self) -> Optional[tuple]:
        """获取下一个数据包"""
        if self.packet_queue:
            return self.packet_queue.pop(0)
        return None
    
    def get_status(self) -> dict:
        """获取组件状态"""
        status = super().get_status()
        status.update({
            "interface": self.interface,
            "filter": self.bpf_filter,
            "offline_file": self.offline_file,
            "mode": self.mode,
            "is_capturing": self.pcap is not None,
            "queue_size": len(self.packet_queue),
            "queue_max_size": self.max_queue_size,
            "stats": self.stats.copy()
        })
        return status