import time
import numpy as np
from collections import defaultdict
from src.features.base_extractor import BaseFeatureExtractor
from src.features.protocol_specs import (
    get_protocol_spec,
    is_feature_relevant
)
from src.utils.helpers import calculate_entropy
from src.config.config_manager import ConfigManager

class StatFeatureExtractor(BaseFeatureExtractor):
    """
    统计特征提取器，提取网络流量的统计特征
    
    包括基本统计特征和协议专属特征
    """
    
    def __init__(self, config=None):
        super().__init__()
        
        # 初始化配置
        self._init_config(config)
    
    def _init_config(self, config):
        """初始化配置"""
        if config is None:
            from src.config.config_manager import ConfigManager
            config = ConfigManager()
        self.config = config
        
        # 初始化特征元数据
        self._init_feature_metadata()
        
        # 从配置加载启用/禁用的特征
        if self.config:
            self.enabled_features = self.config.get("features.enabled_stat_features", [])
            self.disabled_features = self.config.get("features.disabled_stat_features", [])
        
        # 如果没有显式配置，启用所有特征
        if not self.enabled_features and not self.disabled_features:
            self.enabled_features = list(self.feature_metadata.keys())
            
    def _init_feature_metadata(self):
        """初始化特征元数据"""
        self.feature_metadata = {
            # 基本统计特征
            "packet_count": {
                "type": "numeric",
                "description": "会话中的数据包总数",
                "min": 0,
                "max": float("inf")
            },
            "byte_count": {
                "type": "numeric",
                "description": "会话中的总字节数",
                "min": 0,
                "max": float("inf")
            },
            "flow_duration": {
                "type": "numeric",
                "description": "会话持续时间(秒)",
                "min": 0,
                "max": float("inf")
            },
            "avg_packet_size": {
                "type": "numeric",
                "description": "平均数据包大小",
                "min": 0,
                "max": 65535
            },
            "std_packet_size": {
                "type": "numeric",
                "description": "数据包大小标准差",
                "min": 0,
                "max": 65535
            },
            "min_packet_size": {
                "type": "numeric",
                "description": "最小数据包大小",
                "min": 0,
                "max": 65535
            },
            "max_packet_size": {
                "type": "numeric",
                "description": "最大数据包大小",
                "min": 0,
                "max": 65535
            },
            "bytes_per_second": {
                "type": "numeric",
                "description": "每秒字节数",
                "min": 0,
                "max": float("inf")
            },
            "packets_per_second": {
                "type": "numeric",
                "description": "每秒数据包数",
                "min": 0,
                "max": float("inf")
            },
            
            # 协议特征
            "tcp_syn_count": {
                "type": "numeric",
                "description": "TCP SYN包数量",
                "min": 0,
                "max": float("inf")
            },
            "tcp_ack_count": {
                "type": "numeric",
                "description": "TCP ACK包数量",
                "min": 0,
                "max": float("inf")
            },
            "tcp_fin_count": {
                "type": "numeric",
                "description": "TCP FIN包数量",
                "min": 0,
                "max": float("inf")
            },
            "tcp_rst_count": {
                "type": "numeric",
                "description": "TCP RST包数量",
                "min": 0,
                "max": float("inf")
            },
            "tcp_flag_ratio": {
                "type": "numeric",
                "description": "TCP标志包占比",
                "min": 0,
                "max": 1
            },
            "tcp_packet_ratio": {
                "type": "numeric",
                "description": "TCP包占比",
                "min": 0,
                "max": 1
            },
            
            "udp_packet_ratio": {
                "type": "numeric",
                "description": "UDP包占比",
                "min": 0,
                "max": 1
            },
            
            "icmp_packet_ratio": {
                "type": "numeric",
                "description": "ICMP包占比",
                "min": 0,
                "max": 1
            },
            
            # 载荷特征
            "avg_payload_size": {
                "type": "numeric",
                "description": "平均载荷大小",
                "min": 0,
                "max": 65535
            },
            "payload_entropy": {
                "type": "numeric",
                "description": "载荷熵值",
                "min": 0,
                "max": 8
            },
            "payload_size_std": {
                "type": "numeric",
                "description": "载荷大小标准差",
                "min": 0,
                "max": 65535
            },
            
            # 端口特征
            "src_port_entropy": {
                "type": "numeric",
                "description": "源端口熵值",
                "min": 0,
                "max": 16
            },
            "dst_port_entropy": {
                "type": "numeric",
                "description": "目标端口熵值",
                "min": 0,
                "max": 16
            },
            
            # 方向特征
            "outbound_packet_ratio": {
                "type": "numeric",
                "description": "出站包占比",
                "min": 0,
                "max": 1
            },
            "inbound_packet_ratio": {
                "type": "numeric",
                "description": "入站包占比",
                "min": 0,
                "max": 1
            }
        }
    
    def get_feature_names(self):
        """获取所有可能的特征名称"""
        return list(self.feature_metadata.keys())
    
    def extract_features(self, packet, session):
        """
        从单个数据包和会话中提取统计特征
        
        参数:
            packet: 数据包对象
            session: 会话对象
            
        返回:
            特征字典
        """
        features = {}
        
        if not packet or not session:
            return features
            
        # 获取协议信息
        ip_layer = packet.get("ip")
        transport_layer = packet.get("transport")
        
        if not ip_layer or not transport_layer:
            return features
            
        protocol_num = ip_layer.get("protocol", -1)
        
        # 只提取与当前协议相关的特征
        relevant_features = [
            f for f in self.get_enabled_features()
            if is_feature_relevant(protocol_num, f)
        ]
        
        # 基本包特征
        if "packet_size" in relevant_features:
            features["packet_size"] = packet.get("length", 0)
            
        # 载荷特征
        payload = packet.get("payload", b"")
        if "payload_size" in relevant_features:
            features["payload_size"] = len(payload)
        if "payload_entropy" in relevant_features and len(payload) > 0:
            features["payload_entropy"] = calculate_entropy(payload)
        if "has_payload" in relevant_features:
            features["has_payload"] = 1 if len(payload) > 0 else 0
        
        # TCP特征
        if protocol_num == 6 and isinstance(transport_layer, dict):  # TCP
            if "tcp_flags" in relevant_features:
                features["tcp_flags"] = transport_layer.get("flags", 0)
            if "tcp_flag_syn" in relevant_features:
                features["tcp_flag_syn"] = 1 if (transport_layer.get("flags", 0) & 0x02) else 0
            if "tcp_flag_ack" in relevant_features:
                features["tcp_flag_ack"] = 1 if (transport_layer.get("flags", 0) & 0x10) else 0
            if "tcp_flag_fin" in relevant_features:
                features["tcp_flag_fin"] = 1 if (transport_layer.get("flags", 0) & 0x01) else 0
            if "tcp_flag_rst" in relevant_features:
                features["tcp_flag_rst"] = 1 if (transport_layer.get("flags", 0) & 0x04) else 0
            if "window_size" in relevant_features and "winsize" in transport_layer:
                features["window_size"] = transport_layer.get("winsize", 0)
        
        # ICMP特征
        if protocol_num in (1, 58) and isinstance(transport_layer, dict):  # ICMP或ICMPv6
            if "icmp_type" in relevant_features:
                features["icmp_type"] = transport_layer.get("type", -1)
            if "icmp_code" in relevant_features and "code" in transport_layer:
                features["icmp_code"] = transport_layer.get("code", -1)
        
        return features
    
    def extract_features_from_session(self, session):
        """
        从整个会话中提取统计特征
        
        参数:
            session: 会话对象
            
        返回:
            特征字典
        """
        features = {}
        
        if not session or not session.packets or len(session.packets) == 0:
            # 返回零值特征而不是空特征
            for feature_name in self.get_enabled_features():
                features[feature_name] = 0
            return features
            
        # 获取协议信息
        protocol_num = None
        # 尝试从session获取协议号
        if hasattr(session, 'protocol'):
            protocol_num = session.protocol
        else:
            # 从第一个包中获取协议信息
            first_packet = session.packets[0] if session.packets else None
            if first_packet and "ip" in first_packet:
                protocol_num = first_packet["ip"].get("protocol")
        
        if protocol_num is None:
            # 返回零值特征而不是空特征
            for feature_name in self.get_enabled_features():
                features[feature_name] = 0
            return features
            
        spec = get_protocol_spec(protocol_num)
        protocol_name = spec["name"]
        
        # 只提取与当前协议相关的特征
        relevant_features = [
            f for f in self.get_enabled_features()
            if is_feature_relevant(protocol_num, f)
        ]
        
        # 基本会话特征
        if "packet_count" in relevant_features:
            features["packet_count"] = len(session.packets)
        if "byte_count" in relevant_features:
            features["byte_count"] = sum(packet.get("length", 0) for packet in session.packets)
        
        # 时间特征
        timestamps = [packet.get("timestamp", 0) for packet in session.packets if "timestamp" in packet]
        if len(timestamps) >= 2 and "flow_duration" in relevant_features:
            flow_duration = max(timestamps) - min(timestamps)
            features["flow_duration"] = flow_duration if flow_duration > 0 else 0.001  # 避免除以零
        
        # 数据包大小特征
        packet_sizes = [packet.get("length", 0) for packet in session.packets]
        if packet_sizes:
            if "avg_packet_size" in relevant_features:
                features["avg_packet_size"] = np.mean(packet_sizes)
            if "std_packet_size" in relevant_features and len(packet_sizes) > 1:
                features["std_packet_size"] = np.std(packet_sizes)
            if "min_packet_size" in relevant_features:
                features["min_packet_size"] = min(packet_sizes)
            if "max_packet_size" in relevant_features:
                features["max_packet_size"] = max(packet_sizes)
        else:
            # 如果没有数据包，填充默认值
            for feature_name in ["avg_packet_size", "std_packet_size", "min_packet_size", "max_packet_size"]:
                if feature_name in relevant_features:
                    features[feature_name] = 0
        
        # 速率特征
        if "flow_duration" in features and features["flow_duration"] > 0:
            if "bytes_per_second" in relevant_features:
                features["bytes_per_second"] = features["byte_count"] / features["flow_duration"]
            if "packets_per_second" in relevant_features:
                features["packets_per_second"] = features["packet_count"] / features["flow_duration"]
        else:
            # 如果没有持续时间，填充默认值
            for feature_name in ["bytes_per_second", "packets_per_second"]:
                if feature_name in relevant_features:
                    features[feature_name] = 0
        
        # 协议特定特征
        tcp_packets = [p for p in session.packets if p.get("ip", {}).get("protocol") == 6]
        udp_packets = [p for p in session.packets if p.get("ip", {}).get("protocol") == 17]
        icmp_packets = [p for p in session.packets if p.get("ip", {}).get("protocol") in (1, 58)]
        
        total_packets = len(session.packets)
        
        # TCP特征
        if tcp_packets:
            if "tcp_syn_count" in relevant_features:
                syn_count = sum(1 for p in tcp_packets 
                              if p.get("transport", {}).get("flags", 0) & 0x02)
                features["tcp_syn_count"] = syn_count
                
            if "tcp_ack_count" in relevant_features:
                ack_count = sum(1 for p in tcp_packets 
                              if p.get("transport", {}).get("flags", 0) & 0x10)
                features["tcp_ack_count"] = ack_count
                
            if "tcp_fin_count" in relevant_features:
                fin_count = sum(1 for p in tcp_packets 
                              if p.get("transport", {}).get("flags", 0) & 0x01)
                features["tcp_fin_count"] = fin_count
                
            if "tcp_rst_count" in relevant_features:
                rst_count = sum(1 for p in tcp_packets 
                              if p.get("transport", {}).get("flags", 0) & 0x04)
                features["tcp_rst_count"] = rst_count
                
            if "tcp_flag_ratio" in relevant_features:
                flag_packets = sum(1 for p in tcp_packets 
                                 if p.get("transport", {}).get("flags", 0) != 0)
                features["tcp_flag_ratio"] = flag_packets / len(tcp_packets) if tcp_packets else 0
                
            if "tcp_packet_ratio" in relevant_features and total_packets > 0:
                features["tcp_packet_ratio"] = len(tcp_packets) / total_packets
        
        # UDP特征
        if "udp_packet_ratio" in relevant_features and total_packets > 0:
            features["udp_packet_ratio"] = len(udp_packets) / total_packets
            
        # ICMP特征
        if "icmp_packet_ratio" in relevant_features and total_packets > 0:
            features["icmp_packet_ratio"] = len(icmp_packets) / total_packets
        
        # 载荷特征
        payloads = [p.get("payload", b"") for p in session.packets]
        payload_sizes = [len(p) for p in payloads]
        
        if payload_sizes:
            if "avg_payload_size" in relevant_features:
                features["avg_payload_size"] = np.mean(payload_sizes)
            if "payload_size_std" in relevant_features and len(payload_sizes) > 1:
                features["payload_size_std"] = np.std(payload_sizes)
        else:
            # 如果没有载荷，填充默认值
            for feature_name in ["avg_payload_size", "payload_size_std"]:
                if feature_name in relevant_features:
                    features[feature_name] = 0
        
        if payloads and any(p for p in payloads if len(p) > 0):
            if "payload_entropy" in relevant_features:
                # 计算所有载荷的组合熵
                combined_payload = b"".join(p for p in payloads if len(p) > 0)
                features["payload_entropy"] = calculate_entropy(combined_payload) if combined_payload else 0
        else:
            if "payload_entropy" in relevant_features:
                features["payload_entropy"] = 0
        
        # 端口特征
        src_ports = [p.get("src_port", 0) for p in session.packets if "src_port" in p]
        dst_ports = [p.get("dst_port", 0) for p in session.packets if "dst_port" in p]
        
        if src_ports and "src_port_entropy" in relevant_features:
            features["src_port_entropy"] = calculate_entropy(src_ports)
        elif "src_port_entropy" in relevant_features:
            features["src_port_entropy"] = 0
            
        if dst_ports and "dst_port_entropy" in relevant_features:
            features["dst_port_entropy"] = calculate_entropy(dst_ports)
        elif "dst_port_entropy" in relevant_features:
            features["dst_port_entropy"] = 0
        
        # 方向特征
        if hasattr(session, 'direction') and total_packets > 0:
            if "outbound_packet_ratio" in relevant_features:
                outbound_count = sum(1 for p in session.packets 
                                   if getattr(session, 'direction', '') == 'outbound')
                features["outbound_packet_ratio"] = outbound_count / total_packets
                
            if "inbound_packet_ratio" in relevant_features:
                inbound_count = sum(1 for p in session.packets 
                                  if getattr(session, 'direction', '') == 'inbound')
                features["inbound_packet_ratio"] = inbound_count / total_packets
        else:
            # 如果没有方向信息，填充默认值
            for feature_name in ["outbound_packet_ratio", "inbound_packet_ratio"]:
                if feature_name in relevant_features:
                    features[feature_name] = 0
        
        # 确保所有启用的特征都有值
        for feature_name in relevant_features:
            if feature_name not in features:
                features[feature_name] = 0
                
        return features