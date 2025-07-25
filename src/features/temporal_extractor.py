import time
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
    
    def __init__(self):
        super().__init__()
        
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
        
        # 维护每个会话的时序数据
        self.session_temporal_data = {}  # {session_id: {window_type: deque of packets}}
    
    def _init_feature_metadata(self):
        """初始化特征元数据"""
        # 为每个窗口类型创建特征元数据
        for window_type in self.window_configs.keys():
            window_size = self.window_configs[window_type]["size"]
            
            self.feature_metadata[f"{window_type}_window_packet_rate"] = {
                "type": "numeric",
                "description": f"{window_size}秒窗口内的数据包速率(包/秒)"
            }
            
            self.feature_metadata[f"{window_type}_window_byte_rate"] = {
                "type": "numeric",
                "description": f"{window_size}秒窗口内的字节速率(字节/秒)"
            }
            
            self.feature_metadata[f"{window_type}_window_packet_size_mean"] = {
                "type": "numeric",
                "description": f"{window_size}秒窗口内的平均数据包大小"
            }
            
            self.feature_metadata[f"{window_type}_window_packet_size_std"] = {
                "type": "numeric",
                "description": f"{window_size}秒窗口内的数据包大小标准差"
            }
            
            self.feature_metadata[f"{window_type}_window_inter_arrival_mean"] = {
                "type": "numeric",
                "description": f"{window_size}秒窗口内的平均到达间隔(秒)"
            }
            
            self.feature_metadata[f"{window_type}_window_inter_arrival_std"] = {
                "type": "numeric",
                "description": f"{window_size}秒窗口内的到达间隔标准差"
            }
            
            self.feature_metadata[f"{window_type}_window_burst_count"] = {
                "type": "numeric",
                "description": f"{window_size}秒窗口内的流量突发次数"
            }
            
            self.feature_metadata[f"{window_type}_window_burst_duration_mean"] = {
                "type": "numeric",
                "description": f"{window_size}秒窗口内的平均突发持续时间(秒)"
            }
        
        # 跨窗口特征
        self.feature_metadata["packet_rate_trend"] = {
            "type": "numeric",
            "description": "数据包速率趋势（斜率），正为上升，负为下降"
        }
        
        self.feature_metadata["byte_rate_trend"] = {
            "type": "numeric",
            "description": "字节速率趋势（斜率），正为上升，负为下降"
        }
        
        self.feature_metadata["packet_size_variation"] = {
            "type": "numeric",
            "description": "数据包大小的变化率"
        }
        
        self.feature_metadata["inter_arrival_variation"] = {
            "type": "numeric",
            "description": "数据包到达间隔的变化率"
        }
    
    def _get_window_packets(self, session_id, window_type, packet):
        """
        获取指定会话和窗口类型的数据包队列，并添加新数据包
        
        参数:
            session_id: 会话ID
            window_type: 窗口类型 ("short", "medium", "long")
            packet: 新数据包
            
        返回:
            窗口内的数据包队列
        """
        # 初始化会话时序数据
        if session_id not in self.session_temporal_data:
            self.session_temporal_data[session_id] = {}
        
        # 初始化窗口数据包队列
        if window_type not in self.session_temporal_data[session_id]:
            self.session_temporal_data[session_id][window_type] = deque()
        
        window_packets = self.session_temporal_data[session_id][window_type]
        window_size = self.window_configs[window_type]["size"]
        packet_time = packet.get("timestamp", time.time())
        
        # 移除窗口外的数据包
        while window_packets and (packet_time - window_packets[0].get("timestamp", 0) > window_size):
            window_packets.popleft()
        
        # 添加新数据包
        window_packets.append(packet)
        
        return window_packets
    
    def _calculate_burst_features(self, timestamps, inter_arrivals):
        """
        计算流量突发特征
        
        参数:
            timestamps: 时间戳列表
            inter_arrivals: 到达间隔列表
            
        返回:
            包含突发次数和平均突发持续时间的字典
        """
        if len(timestamps) < 2:
            return {"burst_count": 0, "burst_duration_mean": 0}
        
        # 计算突发阈值（使用到达间隔的平均值的1/2）
        mean_inter_arrival = np.mean(inter_arrivals) if inter_arrivals else 0
        burst_threshold = mean_inter_arrival / 2 if mean_inter_arrival > 0 else 0.1
        
        # 检测突发
        bursts = []
        current_burst_start = timestamps[0]
        in_burst = True
        
        for i, inter in enumerate(inter_arrivals):
            if inter < burst_threshold and in_burst:
                # 持续在突发中
                continue
            elif inter >= burst_threshold and in_burst:
                # 突发结束
                current_burst_end = timestamps[i]
                bursts.append({
                    "start": current_burst_start,
                    "end": current_burst_end,
                    "duration": current_burst_end - current_burst_start
                })
                in_burst = False
            elif inter < burst_threshold and not in_burst:
                # 突发开始
                current_burst_start = timestamps[i]
                in_burst = True
        
        # 检查最后是否在突发中
        if in_burst:
            bursts.append({
                "start": current_burst_start,
                "end": timestamps[-1],
                "duration": timestamps[-1] - current_burst_start
            })
        
        # 计算突发特征
        burst_count = len(bursts)
        burst_duration_mean = np.mean([b["duration"] for b in bursts]) if bursts else 0
        
        return {
            "burst_count": burst_count,
            "burst_duration_mean": burst_duration_mean
        }
    
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
        
        if len(short_packets) < 2 or len(medium_packets) < 2:
            return features
        
        # 计算短窗口特征
        short_features = self._calculate_window_features(
            short_packets, "short", short_window["size"]
        )
        
        # 计算中窗口特征
        medium_features = self._calculate_window_features(
            medium_packets, "medium", medium_window["size"]
        )
        
        # 计算速率趋势（中窗口 / 短窗口 - 1）
        if short_features["short_window_packet_rate"] > 0:
            features["packet_rate_trend"] = (
                medium_features["medium_window_packet_rate"] / 
                short_features["short_window_packet_rate"] - 1
            )
        
        if short_features["short_window_byte_rate"] > 0:
            features["byte_rate_trend"] = (
                medium_features["medium_window_byte_rate"] / 
                short_features["short_window_byte_rate"] - 1
            )
        
        # 计算数据包大小变化率
        if short_features["short_window_packet_size_mean"] > 0:
            features["packet_size_variation"] = (
                medium_features["medium_window_packet_size_mean"] / 
                short_features["short_window_packet_size_mean"] - 1
            )
        
        # 计算到达间隔变化率
        if short_features["short_window_inter_arrival_mean"] > 0:
            features["inter_arrival_variation"] = (
                medium_features["medium_window_inter_arrival_mean"] / 
                short_features["short_window_inter_arrival_mean"] - 1
            )
        
        return features
    
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
            
            # 添加所有数据包（会自动过滤窗口外的）
            for packet in session.packets:
                self._get_window_packets(session_id, window_type, packet)
            
            # 获取窗口内的数据包
            window_packets = self.session_temporal_data[session_id][window_type]
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
        
        return features
    
    def cleanup_old_sessions(self, max_age=3600):
        """
        清理旧会话的时序数据
        
        参数:
            max_age: 会话最大保留时间(秒)，默认1小时
        """
        current_time = time.time()
        old_sessions = []
        
        # 找出旧会话
        for session_id, window_data in self.session_temporal_data.items():
            # 检查会话中最新的数据包时间
            latest_time = 0
            for window_packets in window_data.values():
                if window_packets:
                    packet_times = [p.get("timestamp", 0) for p in window_packets]
                    if packet_times:
                        current_latest = max(packet_times)
                        if current_latest > latest_time:
                            latest_time = current_latest
            
            # 如果会话超过max_age没有活动，标记为旧会话
            if current_time - latest_time > max_age:
                old_sessions.append(session_id)
        
        # 清理旧会话
        for session_id in old_sessions:
            del self.session_temporal_data[session_id]
        
        if old_sessions:
            self.logger.debug(f"已清理 {len(old_sessions)} 个旧会话的时序数据")
    
    def start(self):
        """启动时序特征提取器，包括定期清理任务"""
        if self._is_running:
            self.logger.warning("时序特征提取器已在运行中")
            return
            
        super().start()
        
        # 启动定期清理线程
        def cleanup_loop():
            while self._is_running:
                self.cleanup_old_sessions()
                # 每30分钟清理一次
                for _ in range(30):
                    if not self._is_running:
                        break
                    time.sleep(60)
        
        self._processing_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._processing_thread.start()
        
        self.logger.info("时序特征提取器已启动")
    
    def stop(self):
        """停止时序特征提取器"""
        if not self._is_running:
            return
            
        super().stop()
        
        # 停止清理线程
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5)
            if self._processing_thread.is_alive():
                self.logger.warning("时序特征提取器清理线程未能正常终止")
        
        # 清空时序数据
        self.session_temporal_data.clear()
        
        self.logger.info("时序特征提取器已停止")
