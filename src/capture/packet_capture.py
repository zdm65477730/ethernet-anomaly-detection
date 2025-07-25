import pcapy
import dpkt
import time
import threading
import logging
from queue import Queue, Empty
from src.system.base_component import BaseComponent
from src.utils.logger import get_logger

class PacketCapture(BaseComponent):
    """基于pcapy的网络数据包捕获组件，负责实时抓包并解析协议"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("packet_capture")
        self.interface = None  # 网络接口
        self.bpf_filter = None  # BPF过滤规则
        self.pcap = None  # pcapy捕获对象
        self.packet_queue = Queue(maxsize=10000)  # 数据包队列，防止阻塞
        self._capture_thread = None  # 抓包线程
        self._is_capturing = False  # 抓包状态标志
        
        # 协议类型映射
        self.protocol_map = {
            1: "icmp",
            6: "tcp",
            17: "udp"
        }

    def set_interface(self, interface):
        """设置网络接口"""
        self.interface = interface
        return self

    def set_filter(self, bpf_filter):
        """设置BPF过滤规则"""
        self.bpf_filter = bpf_filter
        return self

    def get_next_packet(self, timeout=1):
        """从队列获取下一个数据包"""
        try:
            return self.packet_queue.get(timeout=timeout)
        except Empty:
            return None

    def _packet_handler(self, header, data):
        """pcapy回调函数，处理捕获的数据包"""
        if not self._is_running or not self._is_capturing:
            return
            
        try:
            # 解析以太网帧
            eth = dpkt.ethernet.Ethernet(data)
            
            # 解析IP包
            if not isinstance(eth.data, dpkt.ip.IP):
                self.logger.debug("非IP数据包，跳过")
                return
                
            ip = eth.data
            
            # 构建数据包信息字典
            packet_info = {
                "timestamp": header.getts()[0] + header.getts()[1] / 1e6,  # 时间戳
                "length": len(data),  # 包长度
                "ethernet": {
                    "src": dpkt.utils.mac_to_str(eth.src),
                    "dst": dpkt.utils.mac_to_str(eth.dst),
                    "type": eth.type
                },
                "ip": {
                    "src": dpkt.utils.inet_to_str(ip.src),
                    "dst": dpkt.utils.inet_to_str(ip.dst),
                    "protocol": ip.p,
                    "protocol_name": self.protocol_map.get(ip.p, f"unknown({ip.p})"),
                    "len": ip.len,
                    "ttl": ip.ttl
                }
            }
            
            # 解析传输层协议
            if ip.p == dpkt.ip.IP_PROTO_TCP:
                tcp = ip.data
                packet_info["transport"] = {
                    "src_port": tcp.sport,
                    "dst_port": tcp.dport,
                    "flags": tcp.flags,
                    "seq": tcp.seq,
                    "ack": tcp.ack,
                    "winsize": tcp.win
                }
                packet_info["payload"] = tcp.data
                
            elif ip.p == dpkt.ip.IP_PROTO_UDP:
                udp = ip.data
                packet_info["transport"] = {
                    "src_port": udp.sport,
                    "dst_port": udp.dport,
                    "length": udp.len
                }
                packet_info["payload"] = udp.data
                
            elif ip.p == dpkt.ip.IP_PROTO_ICMP:
                icmp = ip.data
                packet_info["transport"] = {
                    "type": icmp.type,
                    "code": icmp.code
                }
                packet_info["payload"] = icmp.data
            
            # 将解析后的数据包放入队列
            if not self.packet_queue.full():
                self.packet_queue.put(packet_info)
            else:
                self.logger.warning("数据包队列已满，丢弃数据包")
                
        except Exception as e:
            self.logger.error(f"解析数据包出错: {str(e)}", exc_info=True)

    def _capture_loop(self):
        """抓包循环"""
        try:
            # 打开网络接口
            if not self.interface:
                # 如果未指定接口，使用第一个可用接口
                devs = pcapy.findalldevs()
                if not devs:
                    self.logger.error("未找到可用网络接口")
                    return
                self.interface = devs[0]
                self.logger.info(f"未指定接口，使用默认接口: {self.interface}")
            
            # 打开接口，设置缓冲区大小和超时
            self.pcap = pcapy.open_live(
                self.interface,
                65536,  # 最大包长度
                1,      # 混杂模式
                100     # 超时时间(ms)
            )
            
            # 设置过滤规则
            if self.bpf_filter:
                try:
                    self.pcap.setfilter(self.bpf_filter)
                    self.logger.info(f"已设置BPF过滤规则: {self.bpf_filter}")
                except Exception as e:
                    self.logger.error(f"设置过滤规则失败: {str(e)}")
                    return
            
            self._is_capturing = True
            self.logger.info(f"开始在接口 {self.interface} 上捕获数据包")
            
            # 开始捕获
            while self._is_running and self._is_capturing:
                self.pcap.dispatch(100, self._packet_handler)  # 一次处理100个包
                
        except Exception as e:
            self.logger.error(f"抓包循环出错: {str(e)}", exc_info=True)
        finally:
            self._is_capturing = False
            self.pcap = None
            self.logger.info("抓包循环已停止")

    def start(self):
        """启动抓包组件"""
        if self._is_running:
            self.logger.warning("抓包组件已在运行中")
            return
            
        super().start()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        self.logger.info("抓包组件已启动")

    def stop(self):
        """停止抓包组件"""
        if not self._is_running:
            return
            
        self._is_capturing = False
        super().stop()
        
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=5)
            if self._capture_thread.is_alive():
                self.logger.warning("抓包线程未能正常终止")
        
        self.logger.info("抓包组件已停止")

    def get_status(self):
        """获取组件状态"""
        status = super().get_status()
        status.update({
            "interface": self.interface,
            "filter": self.bpf_filter,
            "is_capturing": self._is_capturing,
            "queue_size": self.packet_queue.qsize(),
            "queue_max_size": self.packet_queue.maxsize
        })
        return status
    