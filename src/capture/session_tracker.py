import time
import threading
from typing import Dict, Optional, Any
from src.system.base_component import BaseComponent
from src.utils.logger import get_logger
from src.capture.session import Session

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

    def process_packet(self, packet_data):
        """处理数据包，更新会话状态"""
        if not self._is_running:
            return None
            
        try:
            # packet_data是从PacketCapture.get_next_packet()获取的数据
            # 它应该是已经解析后的字典格式
            packet = packet_data
                
            ip = packet.get("ip")
            transport = packet.get("transport")
            
            if not ip:
                self.logger.debug("忽略非IP数据包")
                return None
                
            self.logger.debug(f"处理数据包: IP={ip}, Transport={transport}")
                
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
                    # 确保端口也交换
                    if src_port is not None and dst_port is not None:
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

    def _session_cleanup_loop(self):
        """会话清理循环"""
        while self._is_running:
            try:
                current_time = time.time()
                
                # 清理过期会话
                expired_sessions = []
                for session_id, session in self.sessions.items():
                    if current_time - session.last_activity > self.session_timeout:
                        expired_sessions.append(session_id)
                
                # 处理过期会话
                for session_id in expired_sessions:
                    session = self.sessions.pop(session_id, None)
                    if session:
                        self.logger.debug(f"会话 {session_id} 已超时，正在处理...")
                        # 将会话数据发送给回调函数（流量分析器）
                        if self.data_callback:
                            try:
                                # 准备会话数据
                                session_data = {
                                    'session_id': session_id,
                                    'src_ip': session.src_ip,
                                    'dst_ip': session.dst_ip,
                                    'src_port': session.src_port,
                                    'dst_port': session.dst_port,
                                    'protocol': session.protocol,
                                    'start_time': session.start_time,
                                    'end_time': current_time,
                                    'packet_count': session.packet_count,
                                    'byte_count': session.byte_count,
                                    'flow_duration': current_time - session.start_time
                                }
                                
                                # 发送数据
                                self.data_callback(session_data)
                                self.logger.debug(f"会话 {session_id} 数据已发送到分析器")
                            except Exception as e:
                                self.logger.error(f"发送会话数据失败: {e}")
                        else:
                            self.logger.warning("未设置数据回调函数，会话数据未发送")
                        
                        self.session_count -= 1
                
                # 如果有会话被处理，记录日志
                if expired_sessions:
                    self.logger.info(f"处理了 {len(expired_sessions)} 个过期会话")
                
                # 等待下次清理
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"会话清理循环出错: {e}")
                time.sleep(1)

    def get_next_updated_session(self, timeout=1):
        """获取下一个更新的会话"""
        if self._session_queue:
            return self._session_queue.pop(0)
        return None

    def start(self):
        """启动会话跟踪器"""
        if self._is_running:
            self.logger.warning("会话跟踪器已在运行中")
            return True
            
        super().start()
        
        # 启动会话清理线程
        self._cleanup_thread = threading.Thread(target=self._session_cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        
        self.logger.info("会话跟踪器已启动")
        return True

    def stop(self):
        """停止会话跟踪器"""
        if not self._is_running:
            return True
            
        self.logger.info("正在停止会话跟踪器...")
        
        # 手动刷新所有会话
        self._flush_all_sessions()
        
        super().stop()
        
        # 停止清理线程
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
            if self._cleanup_thread.is_alive():
                self.logger.warning("会话清理线程未能正常终止")
        
        self.logger.info(f"会话跟踪器已停止，共清理{self.session_count}个会话")
        self.session_count = 0
        return True

    def _flush_all_sessions(self):
        """手动刷新所有会话"""
        try:
            current_time = time.time()
            session_ids = list(self.sessions.keys())
            
            self.logger.info(f"正在刷新 {len(session_ids)} 个会话")
            
            # 处理所有会话
            for session_id in session_ids:
                session = self.sessions.pop(session_id, None)
                if session:
                    # 将会话数据发送给回调函数（流量分析器）
                    if self.data_callback:
                        try:
                            # 准备会话数据
                            session_data = {
                                'session_id': session_id,
                                'src_ip': session.src_ip,
                                'dst_ip': session.dst_ip,
                                'src_port': session.src_port,
                                'dst_port': session.dst_port,
                                'protocol': session.protocol,
                                'start_time': session.start_time,
                                'end_time': current_time,
                                'packet_count': session.packet_count,
                                'byte_count': session.byte_count,
                                'flow_duration': current_time - session.start_time
                            }
                            
                            # 发送数据
                            self.data_callback(session_data)
                            self.logger.debug(f"会话 {session_id} 数据已发送到分析器")
                        except Exception as e:
                            self.logger.error(f"发送会话数据失败: {e}")
                    else:
                        self.logger.warning("未设置数据回调函数，会话数据未发送")
                    
                    self.session_count -= 1
                    
        except Exception as e:
            self.logger.error(f"刷新会话时出错: {e}")

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
    
    def set_data_callback(self, callback):
        """
        设置数据回调函数
        
        参数:
            callback: 回调函数，用于处理会话数据
        """
        self._data_callback = callback

    def _process_completed_sessions(self):
        """处理已完成的会话"""
        current_time = time.time()
        completed_sessions = []
        
        with self._lock:
            # 查找超时的会话
            expired_sessions = []
            for session_id, session in self._sessions.items():
                # 如果会话超过30秒没有新数据包，则认为已完成
                if current_time - session.last_activity_time > 30:
                    expired_sessions.append(session_id)
            
            # 处理过期会话
            for session_id in expired_sessions:
                session = self._sessions.pop(session_id, None)
                if session:
                    completed_sessions.append(session)
        
        # 处理完成的会话
        for session in completed_sessions:
            try:
                session_data = session.to_dict()
                self.logger.debug(f"会话 {session.session_id} 已完成，包含 {len(session.packets)} 个数据包")
                
                # 如果设置了回调函数，调用它
                if hasattr(self, '_data_callback') and self._data_callback:
                    try:
                        self._data_callback(session_data)
                    except Exception as e:
                        self.logger.error(f"调用数据回调函数时出错: {e}")
                else:
                    self.logger.debug("未设置数据回调函数，会话数据未被处理")
                    
            except Exception as e:
                self.logger.error(f"处理完成会话时出错: {e}")
