import time
from typing import List, Dict, Any

class Session:
    """网络会话类，用于跟踪和存储单个网络会话的信息"""
    
    def __init__(self, session_id: str, src_ip: str, dst_ip: str, 
                 protocol: str, src_port: int = None, dst_port: int = None):
        self.session_id = session_id
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.protocol = protocol
        self.src_port = src_port
        self.dst_port = dst_port
        
        # 会话统计信息
        self.packets: List[Dict[str, Any]] = []
        self.total_packets = 0
        self.total_bytes = 0
        self.start_time = None
        self.end_time = None
        self.duration = 0.0
        
        # TCP特定统计信息
        self.tcp_syn_count = 0
        self.tcp_ack_count = 0
        self.tcp_fin_count = 0
        self.tcp_rst_count = 0
        
        # 创建时间
        self.created_at = time.time()
        self.last_seen = self.created_at
    
    def add_packet(self, packet: Dict[str, Any]) -> None:
        """
        向会话中添加数据包
        
        Args:
            packet: 数据包字典
        """
        # 更新时间戳
        timestamp = packet.get("timestamp", time.time())
        if self.start_time is None:
            self.start_time = timestamp
        self.end_time = timestamp
        self.duration = self.end_time - self.start_time if self.start_time else 0.0
        
        # 添加数据包
        self.packets.append(packet)
        self.total_packets += 1
        
        # 更新字节计数
        self.total_bytes += packet.get("length", 0)
        
        # 更新TCP标志统计
        if self.protocol == "tcp":
            tcp_layer = packet.get("transport", {})
            if isinstance(tcp_layer, dict):
                flags = tcp_layer.get("flags", 0)
                if flags & 0x02:  # SYN
                    self.tcp_syn_count += 1
                if flags & 0x10:  # ACK
                    self.tcp_ack_count += 1
                if flags & 0x01:  # FIN
                    self.tcp_fin_count += 1
                if flags & 0x04:  # RST
                    self.tcp_rst_count += 1
        
        # 更新最后活动时间
        self.last_seen = time.time()
        
    # 已移除is_expired方法
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将会话转换为字典格式
        
        Returns:
            会话信息字典
        """
        return {
            "session_id": self.session_id,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "protocol": self.protocol,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "total_packets": self.total_packets,
            "total_bytes": self.total_bytes,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "tcp_syn_count": self.tcp_syn_count,
            "tcp_ack_count": self.tcp_ack_count,
            "tcp_fin_count": self.tcp_fin_count,
            "tcp_rst_count": self.tcp_rst_count,
            "created_at": self.created_at,
            "last_seen": self.last_seen,
            # "is_active": not self.is_expired()  # 暂时注释掉，因为已移除该方法
        }
    
    def __repr__(self) -> str:
        return (f"Session(id={self.session_id}, "
                f"{self.src_ip}:{self.src_port} -> {self.dst_ip}:{self.dst_port}, "
                f"protocol={self.protocol}, packets={self.total_packets})")

# 添加用于调试的函数
def debug_session(session):
    """调试会话对象"""
    print(f"Session ID: {session.session_id}")
    print(f"Packets count: {len(session.packets)}")
    print(f"Protocol: {session.protocol}")
    if session.packets:
        first_packet = session.packets[0]
        print(f"First packet IP: {first_packet.get('ip', {})}")
        print(f"First packet transport: {first_packet.get('transport', {})}")