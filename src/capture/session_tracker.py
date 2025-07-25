import time
import threading
import logging
from collections import defaultdict
from src.system.base_component import BaseComponent
from src.utils.logger import get_logger

class Session:
    """会话对象，存储会话相关信息和数据包"""
    
    def __init__(self, session_id, src_ip, dst_ip, protocol, src_port=None, dst_port=None):
        self.session_id = session_id
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.protocol = protocol
        self.src_port = src_port
        self.dst_port = dst_port
        
        self.first_seen = time.time()  # 首次出现时间
        self.last_seen = self.first_seen  # 最后活动时间
        self.packets = []  # 数据包列表
        self.total_packets = 0  # 总数据包数
        self.total_bytes = 0  # 总字节数
        
        # 方向统计
        self.src_to_dst_packets = 0
        self.src_to_dst_bytes = 0
        self.dst_to_src_packets = 0
        self.dst_to_src_bytes = 0

    def add_packet(self, packet):
        """添加数据包到会话"""
        self.last_seen = time.time()
        self.total_packets += 1
        self.total_bytes += packet["length"]
        
        # 判断方向
        if packet["ip"]["src"] == self.src_ip and packet["ip"]["dst"] == self.dst_ip:
            self.src_to_dst_packets += 1
            self.src_to_dst_bytes += packet["length"]
        else:
            self.dst_to_src_packets += 1
            self.dst_to_src_bytes += packet["length"]
        
        # 限制数据包存储数量，只保留最近的1000个包
        self.packets.append(packet)
        if len(self.packets) > 1000:
            self.packets.pop(0)

    def get_duration(self):
        """获取会话持续时间"""
        return self.last_seen - self.first_seen

    def to_dict(self):
        """转换为字典表示"""
        return {
            "session_id": self.session_id,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "protocol": self.protocol,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "duration": self.get_duration(),
            "total_packets": self.total_packets,
            "total_bytes": self.total_bytes,
            "src_to_dst_packets": self.src_to_dst_packets,
            "src_to_dst_bytes": self.src_to_dst_bytes,
            "dst_to_src_packets": self.dst_to_src_packets,
            "dst_to_src_bytes": self.dst_to_src_bytes,
            "current_packet_count": len(self.packets)
        }


class SessionTracker(BaseComponent):
    """会话跟踪器，负责维护网络会话状态"""
    
    def __init__(self, session_timeout=300):
        super().__init__()
        self.logger = get_logger("session_tracker")
        self.session_timeout = session_timeout  # 会话超时时间(秒)，默认5分钟
        
        self.sessions = {}  # 会话字典，key: session_id, value: Session对象
        self.ip_protocol_map = {
            "tcp": 6,
            "udp": 17,
            "icmp": 1
        }
        
        self._cleanup_thread = None  # 会话清理线程
        self._session_queue = []  # 用于传递会话更新的队列

    def _generate_session_id(self, ip_packet, transport_packet=None):
        """生成会话ID"""
        src_ip = ip_packet["src"]
        dst_ip = ip_packet["dst"]
        protocol = ip_packet["protocol_name"]
        
        # 确保IP顺序一致，避免(src,dst)和(dst,src)被视为不同会话
        if src_ip > dst_ip:
            src_ip, dst_ip = dst_ip, src_ip
        
        # 对于TCP和UDP，加入端口信息
        if protocol in ["tcp", "udp"] and transport_packet:
            src_port = transport_packet["src_port"]
            dst_port = transport_packet["dst_port"]
            
            # 确保端口顺序一致
            if src_port > dst_port:
                src_port, dst_port = dst_port, src_port
                
            return f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
        
        # ICMP没有端口，直接使用IP和协议
        elif protocol == "icmp":
            return f"{src_ip}-{dst_ip}-{protocol}"
        
        # 其他协议
        return f"{src_ip}-{dst_ip}-{protocol}"

    def process_packet(self, packet):
        """处理数据包，更新会话状态"""
        if not self._is_running:
            return None
            
        try:
            ip = packet.get("ip")
            transport = packet.get("transport")
            
            if not ip:
                self.logger.debug("忽略非IP数据包")
                return None
                
            # 生成会话ID
            session_id = self._generate_session_id(ip, transport)
            
            # 检查会话是否已存在，不存在则创建
            if session_id not in self.sessions:
                src_port = transport["src_port"] if transport and "src_port" in transport else None
                dst_port = transport["dst_port"] if transport and "dst_port" in transport else None
                
                # 确保IP顺序与会话ID一致
                src_ip, dst_ip = ip["src"], ip["dst"]
                if src_ip > dst_ip:
                    src_ip, dst_ip = dst_ip, src_ip
                    if src_port and dst_port and src_port < dst_port:
                        src_port, dst_port = dst_port, src_port
                
                self.sessions[session_id] = Session(
                    session_id=session_id,
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    protocol=ip["protocol_name"],
                    src_port=src_port,
                    dst_port=dst_port
                )
                self.logger.debug(f"创建新会话: {session_id}")
            
            # 更新会话
            session = self.sessions[session_id]
            session.add_packet(packet)
            
            # 将更新的会话放入队列，供分析器处理
            self._session_queue.append(session)
            
            return session_id
            
        except Exception as e:
            self.logger.error(f"处理数据包时出错: {str(e)}", exc_info=True)
            return None

    def get_session(self, session_id):
        """获取指定会话"""
        return self.sessions.get(session_id)

    def get_recent_sessions(self, limit=100):
        """获取最近活跃的会话"""
        # 按最后活动时间排序
        sorted_sessions = sorted(
            self.sessions.values(),
            key=lambda s: s.last_seen,
            reverse=True
        )
        return sorted_sessions[:limit]

    def _cleanup_expired_sessions(self):
        """清理超时会话"""
        current_time = time.time()
        expired = []
        
        for session_id, session in self.sessions.items():
            if current_time - session.last_seen > self.session_timeout:
                expired.append(session_id)
        
        if expired:
            self.logger.info(f"清理{len(expired)}个超时会话")
            for session_id in expired:
                del self.sessions[session_id]

    def _cleanup_loop(self):
        """会话清理循环"""
        while self._is_running:
            try:
                self._cleanup_expired_sessions()
                # 每分钟检查一次
                for _ in range(60):
                    if not self._is_running:
                        break
                    time.sleep(1)
            except Exception as e:
                self.logger.error(f"会话清理循环出错: {str(e)}", exc_info=True)
                time.sleep(10)

    def get_next_updated_session(self, timeout=1):
        """获取下一个更新的会话"""
        if self._session_queue:
            return self._session_queue.pop(0)
        return None

    def start(self):
        """启动会话跟踪器"""
        if self._is_running:
            self.logger.warning("会话跟踪器已在运行中")
            return
            
        super().start()
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        self.logger.info("会话跟踪器已启动")

    def stop(self):
        """停止会话跟踪器"""
        if not self._is_running:
            return
            
        super().stop()
        
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
            if self._cleanup_thread.is_alive():
                self.logger.warning("会话清理线程未能正常终止")
        
        self.logger.info(f"会话跟踪器已停止，共清理{len(self.sessions)}个会话")
        self.sessions.clear()

    def get_status(self):
        """获取组件状态"""
        status = super().get_status()
        status.update({
            "session_count": len(self.sessions),
            "session_timeout": self.session_timeout,
            "queue_size": len(self._session_queue),
            "recent_sessions": [s.session_id for s in self.get_recent_sessions(5)]
        })
        return status
    