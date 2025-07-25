import time
import numpy as np
from collections import defaultdict
from src.features.base_extractor import BaseFeatureExtractor
from src.features.protocol_specs import (
    get_protocol_spec,
    is_feature_relevant
)
from src.utils.helpers import calculate_entropy

class StatFeatureExtractor(BaseFeatureExtractor):
    """
    统计特征提取器，提取网络流量的统计特征
    
    包括基本统计特征和协议专属特征
    """
    
    def __init__(self):
        super().__init__()
        
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
            # 基本流量特征
            "packet_count": {
                "type": "numeric",
                "description": "会话中的数据包总数"
            },
            "byte_count": {
                "type": "numeric",
                "description": "会话中的总字节数"
            },
            "flow_duration": {
                "type": "numeric",
                "description": "会话持续时间(秒)"
            },
            "packets_per_second": {
                "type": "numeric",
                "description": "每秒数据包数"
            },
            "bytes_per_second": {
                "type": "numeric",
                "description": "每秒字节数"
            },
            
            # 包大小特征
            "packet_size_mean": {
                "type": "numeric",
                "description": "数据包大小的平均值"
            },
            "packet_size_std": {
                "type": "numeric",
                "description": "数据包大小的标准差"
            },
            "packet_size_min": {
                "type": "numeric",
                "description": "最小数据包大小"
            },
            "packet_size_max": {
                "type": "numeric",
                "description": "最大数据包大小"
            },
            "packet_size_median": {
                "type": "numeric",
                "description": "数据包大小的中位数"
            },
            
            # 载荷特征
            "payload_size_mean": {
                "type": "numeric",
                "description": "载荷大小的平均值"
            },
            "payload_size_std": {
                "type": "numeric",
                "description": "载荷大小的标准差"
            },
            "payload_entropy_mean": {
                "type": "numeric",
                "description": "载荷熵的平均值（衡量随机性）"
            },
            "has_payload": {
                "type": "binary",
                "description": "是否包含载荷（1是，0否）"
            },
            
            # TCP专属特征
            "tcp_flags": {
                "type": "categorical",
                "description": "TCP标志位组合"
            },
            "tcp_flag_syn": {
                "type": "binary",
                "description": "是否设置SYN标志位"
            },
            "tcp_flag_ack": {
                "type": "binary",
                "description": "是否设置ACK标志位"
            },
            "tcp_flag_fin": {
                "type": "binary",
                "description": "是否设置FIN标志位"
            },
            "tcp_flag_rst": {
                "type": "binary",
                "description": "是否设置RST标志位"
            },
            "window_size_mean": {
                "type": "numeric",
                "description": "TCP窗口大小的平均值"
            },
            "window_size_std": {
                "type": "numeric",
                "description": "TCP窗口大小的标准差"
            },
            "retransmission_count": {
                "type": "numeric",
                "description": "TCP重传次数"
            },
            "segment_size_mean": {
                "type": "numeric",
                "description": "TCP段大小的平均值"
            },
            
            # UDP专属特征
            "port_range": {
                "type": "numeric",
                "description": "目标端口范围"
            },
            "unique_dest_ports": {
                "type": "numeric",
                "description": "唯一目标端口数量"
            },
            
            # ICMP专属特征
            "icmp_type": {
                "type": "categorical",
                "description": "ICMP消息类型"
            },
            "icmp_code": {
                "type": "categorical",
                "description": "ICMP消息代码"
            },
            "request_response_ratio": {
                "type": "numeric",
                "description": "ICMP请求与响应的比例"
            },
            "echo_request_count": {
                "type": "numeric",
                "description": "ICMP回显请求数量"
            },
            "echo_reply_count": {
                "type": "numeric",
                "description": "ICMP回显响应数量"
            },
            
            # 时间特征
            "inter_arrival_time_mean": {
                "type": "numeric",
                "description": "数据包到达间隔的平均值"
            },
            "inter_arrival_time_std": {
                "type": "numeric",
                "description": "数据包到达间隔的标准差"
            }
        }
    
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
        if protocol_num == 6 and hasattr(transport_layer, "flags"):  # TCP
            if "tcp_flags" in relevant_features:
                features["tcp_flags"] = transport_layer.flags
            if "tcp_flag_syn" in relevant_features:
                features["tcp_flag_syn"] = 1 if (transport_layer.flags & 0x02) else 0
            if "tcp_flag_ack" in relevant_features:
                features["tcp_flag_ack"] = 1 if (transport_layer.flags & 0x10) else 0
            if "tcp_flag_fin" in relevant_features:
                features["tcp_flag_fin"] = 1 if (transport_layer.flags & 0x01) else 0
            if "tcp_flag_rst" in relevant_features:
                features["tcp_flag_rst"] = 1 if (transport_layer.flags & 0x04) else 0
            if "window_size" in relevant_features and hasattr(transport_layer, "winsize"):
                features["window_size"] = transport_layer.winsize
        
        # ICMP特征
        if protocol_num in (1, 58) and hasattr(transport_layer, "type"):  # ICMP或ICMPv6
            if "icmp_type" in relevant_features:
                features["icmp_type"] = transport_layer.type
            if "icmp_code" in relevant_features and hasattr(transport_layer, "code"):
                features["icmp_code"] = transport_layer.code
        
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
            return features
            
        # 获取协议信息
        protocol_num = session.protocol
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
            
            # 计算每秒数据包数和字节数
            if "packets_per_second" in relevant_features:
                features["packets_per_second"] = len(session.packets) / features["flow_duration"]
            if "bytes_per_second" in relevant_features and "byte_count" in features:
                features["bytes_per_second"] = features["byte_count"] / features["flow_duration"]
        
        # 包大小特征
        packet_sizes = [packet.get("length", 0) for packet in session.packets]
        if packet_sizes and "packet_size_mean" in relevant_features:
            features["packet_size_mean"] = np.mean(packet_sizes)
        if len(packet_sizes) >= 2 and "packet_size_std" in relevant_features:
            features["packet_size_std"] = np.std(packet_sizes)
        if packet_sizes and "packet_size_min" in relevant_features:
            features["packet_size_min"] = np.min(packet_sizes)
        if packet_sizes and "packet_size_max" in relevant_features:
            features["packet_size_max"] = np.max(packet_sizes)
        if packet_sizes and "packet_size_median" in relevant_features:
            features["packet_size_median"] = np.median(packet_sizes)
        
        # 载荷特征
        payloads = [packet.get("payload", b"") for packet in session.packets]
        payload_sizes = [len(p) for p in payloads]
        if payload_sizes and "payload_size_mean" in relevant_features:
            features["payload_size_mean"] = np.mean(payload_sizes)
        if len(payload_sizes) >= 2 and "payload_size_std" in relevant_features:
            features["payload_size_std"] = np.std(payload_sizes)
        if any(payloads) and "payload_entropy_mean" in relevant_features:
            entropies = [calculate_entropy(p) for p in payloads if len(p) > 0]
            if entropies:
                features["payload_entropy_mean"] = np.mean(entropies)
        if "has_payload" in relevant_features:
            features["has_payload"] = 1 if any(len(p) > 0 for p in payloads) else 0
        
        # 时间间隔特征
        if len(timestamps) >= 2 and "inter_arrival_time_mean" in relevant_features:
            inter_arrivals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
            features["inter_arrival_time_mean"] = np.mean(inter_arrivals)
        if len(timestamps) >= 3 and "inter_arrival_time_std" in relevant_features:
            inter_arrivals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
            features["inter_arrival_time_std"] = np.std(inter_arrivals)
        
        # TCP专属特征
        if protocol_num == 6:  # TCP
            # TCP标志位统计
            flags_list = [p.get("transport", {}).get("flags", 0) for p in session.packets]
            if flags_list and "tcp_flags" in relevant_features:
                # 统计最常见的标志组合
                flag_counts = defaultdict(int)
                for flag in flags_list:
                    flag_counts[flag] += 1
                features["tcp_flags"] = max(flag_counts, key=flag_counts.get)
            
            # 特定标志位计数
            if "tcp_flag_syn" in relevant_features:
                syn_count = sum(1 for p in session.packets if (p.get("transport", {}).get("flags", 0) & 0x02))
                features["tcp_flag_syn"] = 1 if syn_count > 0 else 0
            if "tcp_flag_ack" in relevant_features:
                ack_count = sum(1 for p in session.packets if (p.get("transport", {}).get("flags", 0) & 0x10))
                features["tcp_flag_ack"] = 1 if ack_count > 0 else 0
            if "tcp_flag_fin" in relevant_features:
                fin_count = sum(1 for p in session.packets if (p.get("transport", {}).get("flags", 0) & 0x01))
                features["tcp_flag_fin"] = 1 if fin_count > 0 else 0
            if "tcp_flag_rst" in relevant_features:
                rst_count = sum(1 for p in session.packets if (p.get("transport", {}).get("flags", 0) & 0x04))
                features["tcp_flag_rst"] = 1 if rst_count > 0 else 0
            
            # 窗口大小统计
            window_sizes = [p.get("transport", {}).get("winsize", 0) for p in session.packets if p.get("transport")]
            if window_sizes and "window_size_mean" in relevant_features:
                features["window_size_mean"] = np.mean(window_sizes)
            if len(window_sizes) >= 2 and "window_size_std" in relevant_features:
                features["window_size_std"] = np.std(window_sizes)
            
            # 重传计数（简化版，实际应基于序列号检测）
            if "retransmission_count" in relevant_features:
                # 这里使用RST标志位作为重传的简单指标
                features["retransmission_count"] = sum(1 for p in session.packets if (p.get("transport", {}).get("flags", 0) & 0x04))
            
            # 段大小统计
            if "segment_size_mean" in relevant_features:
                segment_sizes = [len(p.get("payload", b"")) for p in session.packets]
                if segment_sizes:
                    features["segment_size_mean"] = np.mean(segment_sizes)
        
        # UDP专属特征
        elif protocol_num == 17:  # UDP
            # 端口特征
            dest_ports = [p.get("transport", {}).get("dport", 0) for p in session.packets if p.get("transport")]
            if dest_ports and "port_range" in relevant_features:
                features["port_range"] = max(dest_ports) - min(dest_ports) if len(set(dest_ports)) > 1 else 0
            if dest_ports and "unique_dest_ports" in relevant_features:
                features["unique_dest_ports"] = len(set(dest_ports))
        
        # ICMP专属特征
        elif protocol_num in (1, 58):  # ICMP或ICMPv6
            icmp_types = [p.get("transport", {}).get("type", -1) for p in session.packets if p.get("transport")]
            icmp_codes = [p.get("transport", {}).get("code", -1) for p in session.packets if p.get("transport")]
            
            if icmp_types and "icmp_type" in relevant_features:
                # 最常见的类型
                type_counts = defaultdict(int)
                for t in icmp_types:
                    type_counts[t] += 1
                features["icmp_type"] = max(type_counts, key=type_counts.get)
            
            if icmp_codes and "icmp_code" in relevant_features:
                # 最常见的代码
                code_counts = defaultdict(int)
                for c in icmp_codes:
                    code_counts[c] += 1
                features["icmp_code"] = max(code_counts, key=code_counts.get)
            
            # 回显请求/响应计数
            if "echo_request_count" in relevant_features or "echo_reply_count" in relevant_features:
                echo_requests = 0
                echo_replies = 0
                for p in session.packets:
                    transport = p.get("transport", {})
                    if not transport:
                        continue
                    if protocol_num == 1:  # ICMP
                        if transport.get("type") == 8:  # 回显请求
                            echo_requests += 1
                        elif transport.get("type") == 0:  # 回显响应
                            echo_replies += 1
                    elif protocol_num == 58:  # ICMPv6
                        if transport.get("type") == 128:  # 回显请求
                            echo_requests += 1
                        elif transport.get("type") == 129:  # 回显响应
                            echo_replies += 1
                
                if "echo_request_count" in relevant_features:
                    features["echo_request_count"] = echo_requests
                if "echo_reply_count" in relevant_features:
                    features["echo_reply_count"] = echo_replies
                
                # 请求响应比例
                if "request_response_ratio" in relevant_features:
                    if echo_replies == 0:
                        features["request_response_ratio"] = float('inf') if echo_requests > 0 else 0
                    else:
                        features["request_response_ratio"] = echo_requests / echo_replies
        
        # 填充缺失值为0
        for feature in relevant_features:
            if feature not in features:
                features[feature] = 0
        
        return features
