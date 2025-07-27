import time
import threading
import numpy as np
from collections import deque
from src.features.base_extractor import BaseFeatureExtractor
from src.features.protocol_specs import (
    get_protocol_spec,
    is_feature_relevant
)

class TemporalFeatureExtractor(BaseFeatureExtractor):
    """
    时序特征提取器，提取网络流量的时序特征
    
    基于滑动窗口计算各类时间相关特征
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
        
        # 滑动窗口配置
        self.window_configs = {
            "short": {"size": 10, "step": 2},   # 短窗口：10秒，步长2秒
            "medium": {"size": 60, "step": 10}, # 中窗口：60秒，步长10秒
            "long": {"size": 300, "step": 60}   # 长窗口：300秒，步长60秒
        }
        
        # 初始化特征元数据
        self._init_feature_metadata()
        
        # 从配置加载启用/禁用的特征
        if self.config:
            self.enabled_features = self.config.get("features.enabled_temporal_features", [])
            self.disabled_features = self.config.get("features.disabled_temporal_features", [])
        
        # 如果没有显式配置，启用所有特征
        if not self.enabled_features and not self.disabled_features:
            self.enabled_features = list(self.feature_metadata.keys())
            
        # 初始化滑动窗口
        self._init_windows()
    
    def _init_feature_metadata(self):
        """初始化特征元数据"""
        self.feature_metadata = {}
        
        # 为每个窗口类型初始化特征
        for window_type in self.window_configs.keys():
            self.feature_metadata.update({
                f"{window_type}_window_packet_rate": {
                    "type": "numeric",
                    "description": f"{window_type}窗口数据包速率(包/秒)",
                    "min": 0,
                    "max": float("inf")
                },
                f"{window_type}_window_byte_rate": {
                    "type": "numeric",
                    "description": f"{window_type}窗口字节速率(字节/秒)",
                    "min": 0,
                    "max": float("inf")
                },
                f"{window_type}_window_packet_size_mean": {
                    "type": "numeric",
                    "description": f"{window_type}窗口平均数据包大小",
                    "min": 0,
                    "max": 65535
                },
                f"{window_type}_window_packet_size_std": {
                    "type": "numeric",
                    "description": f"{window_type}窗口数据包大小标准差",
                    "min": 0,
                    "max": 65535
                },
                f"{window_type}_window_inter_arrival_mean": {
                    "type": "numeric",
                    "description": f"{window_type}窗口包到达间隔平均值(秒)",
                    "min": 0,
                    "max": float("inf")
                },
                f"{window_type}_window_inter_arrival_std": {
                    "type": "numeric",
                    "description": f"{window_type}窗口包到达间隔标准差",
                    "min": 0,
                    "max": float("inf")
                },
                f"{window_type}_window_burst_count": {
                    "type": "numeric",
                    "description": f"{window_type}窗口突发次数",
                    "min": 0,
                    "max": float("inf")
                },
                f"{window_type}_window_burst_duration_mean": {
                    "type": "numeric",
                    "description": f"{window_type}窗口平均突发持续时间(秒)",
                    "min": 0,
                    "max": float("inf")
                }
            })
        
        # 趋势特征
        self.feature_metadata.update({
            "packet_rate_trend": {
                "type": "numeric",
                "description": "数据包速率趋势",
                "min": -float("inf"),
                "max": float("inf")
            },
            "byte_rate_trend": {
                "type": "numeric",
                "description": "字节速率趋势",
                "min": -float("inf"),
                "max": float("inf")
            },
            "packet_size_variation": {
                "type": "numeric",
                "description": "数据包大小变化率",
                "min": 0,
                "max": float("inf")
            },
            "inter_arrival_variation": {
                "type": "numeric",
                "description": "包到达间隔变化率",
                "min": 0,
                "max": float("inf")
            }
        })
    
    def get_feature_names(self):
        """获取所有可能的特征名称"""
        return list(self.feature_metadata.keys())
    
    def _init_windows(self):
        """初始化滑动窗口数据结构"""
        self.session_temporal_data = {}  # {session_id: {window_type: deque}}
        
    def _calculate_window_features(self, window_packets, window_type, window_size):
        """
        计算单个窗口的时序特征
        
        参数:
            window_packets: 窗口内的数据包列表
            window_type: 窗口类型
            window_size: 窗口大小(秒)
            
        返回:
            特征字典
        """
        features = {}
        
        if len(window_packets) == 0:
            # 窗口内没有数据包，填充默认值
            features[f"{window_type}_window_packet_rate"] = 0
            features[f"{window_type}_window_byte_rate"] = 0
            features[f"{window_type}_window_packet_size_mean"] = 0
            features[f"{window_type}_window_packet_size_std"] = 0
            features[f"{window_type}_window_inter_arrival_mean"] = 0
            features[f"{window_type}_window_inter_arrival_std"] = 0
            features[f"{window_type}_window_burst_count"] = 0
            features[f"{window_type}_window_burst_duration_mean"] = 0
            return features
        
        # 提取基本信息
        timestamps = [p.get("timestamp", 0) for p in window_packets]
        packet_sizes = [p.get("length", 0) for p in window_packets]
        total_bytes = sum(packet_sizes)
        
        # 计算实际窗口持续时间（最后一个包 - 第一个包）
        actual_duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else window_size
        
        # 数据包速率和字节速率
        features[f"{window_type}_window_packet_rate"] = len(window_packets) / actual_duration
        features[f"{window_type}_window_byte_rate"] = total_bytes / actual_duration
        
        # 数据包大小统计
        features[f"{window_type}_window_packet_size_mean"] = np.mean(packet_sizes)
        features[f"{window_type}_window_packet_size_std"] = np.std(packet_sizes) if len(packet_sizes) > 1 else 0
        
        # 到达间隔统计
        inter_arrivals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))] if len(timestamps) > 1 else []
        if inter_arrivals:
            features[f"{window_type}_window_inter_arrival_mean"] = np.mean(inter_arrivals)
            features[f"{window_type}_window_inter_arrival_std"] = np.std(inter_arrivals) if len(inter_arrivals) > 1 else 0
        else:
            features[f"{window_type}_window_inter_arrival_mean"] = 0
            features[f"{window_type}_window_inter_arrival_std"] = 0
        
        # 突发特征
        burst_features = self._calculate_burst_features(timestamps, inter_arrivals)
        features[f"{window_type}_window_burst_count"] = burst_features["burst_count"]
        features[f"{window_type}_window_burst_duration_mean"] = burst_features["burst_duration_mean"]
        
        return features
    
    def _calculate_burst_features(self, timestamps, inter_arrivals, burst_threshold=0.1):
        """
        计算突发特征
        
        参数:
            timestamps: 时间戳列表
            inter_arrivals: 到达间隔列表
            burst_threshold: 突发阈值(秒)
            
        返回:
            突发特征字典
        """
        if not inter_arrivals:
            return {"burst_count": 0, "burst_duration_mean": 0}
        
        # 识别突发: 连续的小间隔包
        bursts = []
        current_burst = [timestamps[0]]
        
        for i, interval in enumerate(inter_arrivals):
            if interval <= burst_threshold:
                # 属于当前突发
                current_burst.append(timestamps[i+1])
            else:
                # 突发结束
                if len(current_burst) > 1:  # 至少2个包才算突发
                    bursts.append({
                        "start": current_burst[0],
                        "end": current_burst[-1],
                        "duration": current_burst[-1] - current_burst[0],
                        "packet_count": len(current_burst)
                    })
                # 开始新突发
                current_burst = [timestamps[i+1]]
        
        # 处理最后一个突发
        if len(current_burst) > 1:
            bursts.append({
                "start": current_burst[0],
                "end": current_burst[-1],
                "duration": current_burst[-1] - current_burst[0],
                "packet_count": len(current_burst)
            })
        
        # 计算突发特征
        burst_count = len(bursts)
        burst_duration_mean = np.mean([b["duration"] for b in bursts]) if bursts else 0
        
        return {
            "burst_count": burst_count,
            "burst_duration_mean": burst_duration_mean
        }
    
    def _calculate_trend_features(self, session_id):
        """
        计算跨窗口的趋势特征
        
        参数:
            session_id: 会话ID
            
        返回:
            特征字典
        """
        features = {
            "packet_rate_trend": 0,
            "byte_rate_trend": 0,
            "packet_size_variation": 0,
            "inter_arrival_variation": 0
        }
        
        # 检查会话是否有时序数据
        if session_id not in self.session_temporal_data:
            return features
        
        # 我们需要至少两个窗口的数据来计算趋势
        window_types = list(self.window_configs.keys())
        if len(window_types) < 2:
            return features
        
        # 为简化，我们使用短窗口和中窗口的数据来计算趋势
        short_window = self.window_configs["short"]
        medium_window = self.window_configs["medium"]
        
        short_packets = self.session_temporal_data[session_id].get("short", deque())
        medium_packets = self.session_temporal_data[session_id].get("medium", deque())
        
        if len(short_packets) == 0 or len(medium_packets) == 0:
            return features
        
        # 计算短窗口特征
        short_features = self._calculate_window_features(
            list(short_packets), "short", short_window["size"]
        )
        
        # 计算中窗口特征
        medium_features = self._calculate_window_features(
            list(medium_packets), "medium", medium_window["size"]
        )
        
        # 计算趋势（中窗口 - 短窗口）
        features["packet_rate_trend"] = (
            medium_features["medium_window_packet_rate"] - 
            short_features["short_window_packet_rate"]
        )
        
        features["byte_rate_trend"] = (
            medium_features["medium_window_byte_rate"] - 
            short_features["short_window_byte_rate"]
        )
        
        # 计算变化率
        if short_features["short_window_packet_size_mean"] > 0:
            features["packet_size_variation"] = abs(
                medium_features["medium_window_packet_size_mean"] - 
                short_features["short_window_packet_size_mean"]
            ) / short_features["short_window_packet_size_mean"]
        
        if short_features["short_window_inter_arrival_mean"] > 0:
            features["inter_arrival_variation"] = abs(
                medium_features["medium_window_inter_arrival_mean"] - 
                short_features["short_window_inter_arrival_mean"]
            ) / short_features["short_window_inter_arrival_mean"]
        
        return features
    
    def _get_window_packets(self, session_id, window_type, new_packet):
        """
        获取指定窗口内的数据包
        
        参数:
            session_id: 会话ID
            window_type: 窗口类型
            new_packet: 新数据包
            
        返回:
            窗口内的数据包列表
        """
        # 确保会话的时序数据已初始化
        if session_id not in self.session_temporal_data:
            self.session_temporal_data[session_id] = {}
            
        if window_type not in self.session_temporal_data[session_id]:
            self.session_temporal_data[session_id][window_type] = deque()
            
        window_packets = self.session_temporal_data[session_id][window_type]
        
        # 添加新数据包
        if new_packet:
            window_packets.append(new_packet)
            
        # 获取窗口配置
        window_config = self.window_configs[window_type]
        window_size = window_config["size"]
        
        # 移除超出窗口范围的数据包
        if new_packet and "timestamp" in new_packet:
            current_time = new_packet["timestamp"]
            expire_time = current_time - window_size
            
            # 移除过期的数据包
            while window_packets and window_packets[0].get("timestamp", 0) < expire_time:
                window_packets.popleft()
        
        return list(window_packets)
    
    def extract_features(self, packet, session):
        """
        从单个数据包和会话中提取时序特征
        
        参数:
            packet: 数据包对象
            session: 会话对象
            
        返回:
            特征字典
        """
        features = {}
        
        if not packet or not session:
            # 返回零值特征而不是空特征
            for feature_name in self.get_enabled_features():
                features[feature_name] = 0
            return features
            
        session_id = session.session_id
        protocol_num = session.protocol
        
        # 只提取与当前协议相关的特征
        relevant_features = [
            f for f in self.get_enabled_features()
            if is_feature_relevant(protocol_num, f)
        ]
        
        # 为每个窗口类型计算特征
        for window_type in self.window_configs.keys():
            # 获取窗口内的数据包
            window_packets = self._get_window_packets(session_id, window_type, packet)
            window_size = self.window_configs[window_type]["size"]
            
            # 计算窗口特征
            window_features = self._calculate_window_features(
                window_packets, window_type, window_size
            )
            
            # 只保留相关的特征
            for feature_name, value in window_features.items():
                if feature_name in relevant_features:
                    features[feature_name] = value
        
        # 计算趋势特征
        if any(f in relevant_features for f in [
            "packet_rate_trend", "byte_rate_trend",
            "packet_size_variation", "inter_arrival_variation"
        ]):
            trend_features = self._calculate_trend_features(session_id)
            for feature_name, value in trend_features.items():
                if feature_name in relevant_features:
                    features[feature_name] = value
        
        # 确保所有启用的特征都有值
        for feature_name in relevant_features:
            if feature_name not in features:
                features[feature_name] = 0
                
        return features
    
    def extract_features_from_session(self, session):
        """
        从整个会话中提取时序特征
        
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
            
        session_id = session.session_id
        protocol_num = session.protocol
        
        # 只提取与当前协议相关的特征
        relevant_features = [
            f for f in self.get_enabled_features()
            if is_feature_relevant(protocol_num, f)
        ]
        
        # 确保会话的时序数据已初始化
        if session_id not in self.session_temporal_data:
            self.session_temporal_data[session_id] = {}
        
        # 为每个窗口类型添加所有数据包
        for window_type in self.window_configs.keys():
            # 初始化窗口数据包队列
            if window_type not in self.session_temporal_data[session_id]:
                self.session_temporal_data[session_id][window_type] = deque()
            
            window_packets = self.session_temporal_data[session_id][window_type]
            
            # 添加所有数据包到窗口
            for packet in session.packets:
                window_packets.append(packet)
            
            # 获取窗口配置
            window_config = self.window_configs[window_type]
            window_size = window_config["size"]
            
            # 移除超出窗口范围的数据包（基于最后一个包的时间）
            if session.packets and "timestamp" in session.packets[-1]:
                current_time = session.packets[-1]["timestamp"]
                expire_time = current_time - window_size
                
                # 移除过期的数据包
                while window_packets and window_packets[0].get("timestamp", 0) < expire_time:
                    window_packets.popleft()
            
            # 计算窗口特征
            window_features = self._calculate_window_features(
                list(window_packets), window_type, window_size
            )
            
            # 只保留相关的特征
            for feature_name, value in window_features.items():
                if feature_name in relevant_features:
                    features[feature_name] = value
        
        # 计算趋势特征
        if any(f in relevant_features for f in [
            "packet_rate_trend", "byte_rate_trend",
            "packet_size_variation", "inter_arrival_variation"
        ]):
            trend_features = self._calculate_trend_features(session_id)
            for feature_name, value in trend_features.items():
                if feature_name in relevant_features:
                    features[feature_name] = value
        
        # 确保所有启用的特征都有值
        for feature_name in relevant_features:
            if feature_name not in features:
                features[feature_name] = 0
                
        return features