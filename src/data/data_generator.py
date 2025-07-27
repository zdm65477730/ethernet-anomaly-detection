import os
import random
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.utils.logger import get_logger
from src.features.protocol_specs import PROTOCOL_SPECS

class DataGenerator:
    """生成模拟的网络流量数据，用于测试和模型训练"""
    
    def __init__(self):
        self.logger = get_logger("data_generator")
        
        # 协议配置（协议号: (名称, 出现概率)）
        self.protocols = {
            protocol: (PROTOCOL_SPECS[protocol]["name"], prob)
            for protocol, prob in [(6, 0.6), (17, 0.3), (1, 0.1)]
        }
        
        # IP地址池
        self.ip_pool = self._generate_ip_pool(100)  # 生成100个随机IP
        
        # 端口范围
        self.well_known_ports = list(range(1, 1024))
        self.ephemeral_ports = list(range(1024, 65535))
        
        # 异常类型及概率
        self.anomaly_types = {
            "normal": 0.9,                  # 正常流量占90%
            "syn_flood": 0.02,              # SYN Flood攻击
            "port_scan": 0.02,              # 端口扫描
            "udp_amplification": 0.02,      # UDP放大攻击
            "icmp_flood": 0.02,             # ICMP Flood攻击
            "large_payload": 0.01,          # 大 payload 攻击
            "unusual_flags": 0.01           # 异常TCP标志
        }
        
        # 初始化随机种子，保证一定的可重复性
        random.seed(42)
        np.random.seed(42)
    
    def _generate_ip_pool(self, count):
        """生成指定数量的随机IP地址"""
        ip_pool = []
        for _ in range(count):
            ip = ".".join(str(random.randint(1, 254)) for _ in range(4))
            ip_pool.append(ip)
        return ip_pool
    
    def _random_ip(self):
        """随机选择一个IP地址"""
        return random.choice(self.ip_pool)
    
    def _random_port(self, is_source=True):
        """随机选择一个端口号"""
        if is_source:
            # 源端口更多使用临时端口
            return random.choice(self.ephemeral_ports)
        else:
            # 目的端口更多使用知名端口
            return random.choice(self.well_known_ports + self.ephemeral_ports[:1000])
    
    def _random_protocol(self):
        """根据概率随机选择一个协议"""
        protocols = list(self.protocols.keys())
        probabilities = [weight for _, weight in self.protocols.values()]
        return random.choices(protocols, weights=probabilities)[0]
    
    def _random_anomaly_type(self):
        """根据概率随机选择一个异常类型"""
        types = list(self.anomaly_types.keys())
        probabilities = list(self.anomaly_types.values())
        return random.choices(types, weights=probabilities)[0]
    
    def _generate_tcp_flags(self, anomaly_type):
        """生成TCP标志位"""
        if anomaly_type == "syn_flood":
            # SYN Flood 攻击，主要是SYN标志
            return "SYN"
        elif anomaly_type == "unusual_flags":
            # 异常标志组合
            return random.choice(["FIN+URG+PSH", "SYN+FIN", "SYN+RST"])
        else:
            # 正常标志组合
            return random.choice(["SYN", "ACK", "SYN+ACK", "FIN+ACK", "PSH+ACK"])
    
    def _generate_packet_size(self, protocol, anomaly_type):
        """生成数据包大小"""
        if anomaly_type == "large_payload":
            # 大 payload 攻击
            return random.randint(1400, 1500)
        elif protocol == 6:  # TCP
            return random.randint(40, 1500)
        elif protocol == 17:  # UDP
            if anomaly_type == "udp_amplification":
                return random.randint(1000, 1500)
            return random.randint(28, 1500)
        elif protocol == 1:  # ICMP
            if anomaly_type == "icmp_flood":
                return random.randint(64, 128)
            return random.randint(64, 128)
        return random.randint(40, 1500)
    
    def _generate_timestamps(self, num_samples, start_time=None):
        """生成时间戳序列，模拟流量的时间分布"""
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=1)
        
        # 生成基础时间戳（均匀分布）
        base_timestamps = [
            start_time + timedelta(seconds=i * (3600 / num_samples))
            for i in range(num_samples)
        ]
        
        # 添加随机波动，模拟真实网络的不均匀性
        timestamps = []
        for ts in base_timestamps:
            # 正常波动（±5秒）
            jitter = random.uniform(-5, 5)
            adjusted_ts = ts + timedelta(seconds=jitter)
            timestamps.append(adjusted_ts.timestamp())
        
        return sorted(timestamps)
    
    def generate_sample(self, timestamp, anomaly_type=None):
        """生成单个流量样本"""
        # 随机选择异常类型（如果未指定）
        if anomaly_type is None:
            anomaly_type = self._random_anomaly_type()
        
        # 标签：1表示异常，0表示正常
        label = 0 if anomaly_type == "normal" else 1
        
        # 选择协议
        protocol = self._random_protocol()
        
        # 针对端口扫描攻击，强制使用TCP协议
        if anomaly_type == "port_scan":
            protocol = 6
        
        # 针对ICMP Flood攻击，强制使用ICMP协议
        if anomaly_type == "icmp_flood":
            protocol = 1
        
        # 生成源IP和目的IP
        src_ip = self._random_ip()
        dst_ip = self._random_ip()
        
        # 确保源IP和目的IP不同
        while src_ip == dst_ip:
            dst_ip = self._random_ip()
        
        # 生成端口（ICMP没有端口）
        src_port = self._random_port(is_source=True) if protocol != 1 else 0
        dst_port = self._random_port(is_source=False) if protocol != 1 else 0
        
        # 针对端口扫描，使用不同的目的端口
        if anomaly_type == "port_scan":
            dst_port = random.choice(self.well_known_ports)
        
        # 生成包大小
        packet_size = self._generate_packet_size(protocol, anomaly_type)
        
        # 生成TCP标志位（仅TCP协议）
        tcp_flags = self._generate_tcp_flags(anomaly_type) if protocol == 6 else ""
        
        # 生成payload熵
        payload_entropy = random.uniform(1.0, 7.0)
        if anomaly_type == "large_payload":
            payload_entropy = random.uniform(0.1, 2.0)  # 大payload通常熵较低
        
        # 生成会话持续时间（秒）
        session_duration = random.uniform(0.1, 300.0)
        
        # 生成包间隔时间（秒）
        inter_arrival_time = random.uniform(0.001, 5.0)
        if anomaly_type in ["syn_flood", "icmp_flood"]:
            inter_arrival_time = random.uniform(0.0001, 0.01)  # 攻击包间隔更小
        
        # 生成窗口大小（仅TCP）
        window_size = random.randint(1024, 65535) if protocol == 6 else 0
        
        # 生成重传次数
        retransmissions = random.randint(0, 3)
        if anomaly_type in ["syn_flood", "port_scan"]:
            retransmissions = 0  # 攻击通常没有重传
        
        # 生成ICMP类型和代码（仅ICMP）
        icmp_type = random.randint(0, 15) if protocol == 1 else 0
        icmp_code = random.randint(0, 15) if protocol == 1 else 0
        
        # 构建样本字典（原始特征）
        raw_sample = {
            "timestamp": timestamp,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": protocol,
            "protocol_name": PROTOCOL_SPECS.get(protocol, {"name": "unknown"})["name"],
            "packet_size": packet_size,
            "tcp_flags": tcp_flags,
            "payload_entropy": payload_entropy,
            "session_duration": session_duration,
            "inter_arrival_time": inter_arrival_time,
            "window_size": window_size,
            "retransmissions": retransmissions,
            "icmp_type": icmp_type,
            "icmp_code": icmp_code,
            "anomaly_type": anomaly_type,
            "label": label
        }
        
        return raw_sample
    
    def _convert_raw_to_model_features(self, raw_samples):
        """
        将原始样本转换为模型兼容的特征
        
        Args:
            raw_samples: 原始样本列表
            
        Returns:
            包含模型兼容特征的DataFrame
        """
        model_samples = []
        
        for sample in raw_samples:
            # 解析TCP标志
            tcp_flags = sample.get("tcp_flags", "")
            has_syn = 1 if "SYN" in tcp_flags else 0
            has_ack = 1 if "ACK" in tcp_flags else 0
            has_fin = 1 if "FIN" in tcp_flags else 0
            has_rst = 1 if "RST" in tcp_flags else 0
            
            # 计算一些衍生特征
            packet_count = 1  # 简化处理，每个样本视为一个包
            byte_count = sample.get("packet_size", 0)
            flow_duration = sample.get("session_duration", 0)
            avg_packet_size = sample.get("packet_size", 0)
            std_packet_size = 0  # 单个包的标准差为0
            min_packet_size = sample.get("packet_size", 0)
            max_packet_size = sample.get("packet_size", 0)
            
            # 计算速率特征
            bytes_per_second = byte_count / flow_duration if flow_duration > 0 else 0
            packets_per_second = packet_count / flow_duration if flow_duration > 0 else 0
            
            # 构建模型兼容的样本
            model_sample = {
                # 基本统计特征
                "packet_count": packet_count,
                "byte_count": byte_count,
                "flow_duration": flow_duration,
                "avg_packet_size": avg_packet_size,
                "std_packet_size": std_packet_size,
                "min_packet_size": min_packet_size,
                "max_packet_size": max_packet_size,
                "bytes_per_second": bytes_per_second,
                "packets_per_second": packets_per_second,
                
                # 协议特征
                "tcp_syn_count": has_syn,
                "tcp_ack_count": has_ack,
                "tcp_fin_count": has_fin,
                "tcp_rst_count": has_rst,
                "tcp_flag_ratio": (has_syn + has_ack + has_fin + has_rst) / 4 if tcp_flags else 0,
                "tcp_packet_ratio": 1 if sample.get("protocol") == 6 else 0,
                "udp_packet_ratio": 1 if sample.get("protocol") == 17 else 0,
                "icmp_packet_ratio": 1 if sample.get("protocol") == 1 else 0,
                
                # 载荷特征
                "avg_payload_size": sample.get("packet_size", 0),  # 简化处理
                "payload_entropy": sample.get("payload_entropy", 0),
                "payload_size_std": 0,  # 单个包的标准差为0
                
                # 端口特征（简化处理）
                "src_port_entropy": 0,
                "dst_port_entropy": 0,
                
                # 方向特征（简化处理）
                "outbound_packet_ratio": 0.5,
                "inbound_packet_ratio": 0.5,
                
                # 时序特征（简化处理）
                "short_window_packet_rate": packets_per_second,
                "short_window_byte_rate": bytes_per_second,
                "short_window_packet_size_mean": avg_packet_size,
                "short_window_packet_size_std": std_packet_size,
                "short_window_inter_arrival_mean": sample.get("inter_arrival_time", 0),
                "short_window_inter_arrival_std": 0,
                "short_window_burst_count": 0,
                "short_window_burst_duration_mean": 0,
                
                "medium_window_packet_rate": packets_per_second,
                "medium_window_byte_rate": bytes_per_second,
                "medium_window_packet_size_mean": avg_packet_size,
                "medium_window_packet_size_std": std_packet_size,
                "medium_window_inter_arrival_mean": sample.get("inter_arrival_time", 0),
                "medium_window_inter_arrival_std": 0,
                "medium_window_burst_count": 0,
                "medium_window_burst_duration_mean": 0,
                
                "long_window_packet_rate": packets_per_second,
                "long_window_byte_rate": bytes_per_second,
                "long_window_packet_size_mean": avg_packet_size,
                "long_window_packet_size_std": std_packet_size,
                "long_window_inter_arrival_mean": sample.get("inter_arrival_time", 0),
                "long_window_inter_arrival_std": 0,
                "long_window_burst_count": 0,
                "long_window_burst_duration_mean": 0,
                
                # 趋势特征（简化处理）
                "packet_rate_trend": 0,
                "byte_rate_trend": 0,
                "packet_size_variation": 0,
                "inter_arrival_variation": 0,
                
                # 标签
                "label": sample.get("label", 0)
            }
            
            model_samples.append(model_sample)
        
        return pd.DataFrame(model_samples)
    
    def generate(self, num_samples=10000, output_dir="data/raw", split_train_test=True, 
                 test_size=0.2, start_time=None, force=False, generate_model_features=True):
        """
        生成指定数量的模拟流量数据
        
        参数:
            num_samples: 样本数量
            output_dir: 输出目录
            split_train_test: 是否分割训练集和测试集
            test_size: 测试集比例
            start_time: 开始时间
            force: 如果目录已存在，是否强制覆盖
            generate_model_features: 是否生成模型兼容特征
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info(f"开始生成 {num_samples} 个模拟流量样本...")
        
        # 生成时间戳
        timestamps = self._generate_timestamps(num_samples, start_time)
        
        # 生成样本
        raw_samples = []
        for i, ts in enumerate(timestamps):
            if i % 1000 == 0 and i > 0:
                self.logger.info(f"已生成 {i}/{num_samples} 个样本")
            
            try:
                sample = self.generate_sample(ts)
                raw_samples.append(sample)
            except Exception as e:
                self.logger.error(f"生成第 {i} 个样本时出错: {str(e)}")
                raise e
        
        # 转换为DataFrame
        raw_df = pd.DataFrame(raw_samples)
        
        # 显示数据分布
        self.logger.info(f"数据分布:")
        self.logger.info(f"  正常样本: {len(raw_df[raw_df['label'] == 0])} ({len(raw_df[raw_df['label'] == 0])/len(raw_df):.2%})")
        self.logger.info(f"  异常样本: {len(raw_df[raw_df['label'] == 1])} ({len(raw_df[raw_df['label'] == 1])/len(raw_df):.2%})")
        self.logger.info(f"  协议分布: \n{raw_df['protocol_name'].value_counts().to_string()}")
        
        # 保存原始数据
        raw_path = os.path.join(output_dir, "raw_data.csv")
        raw_df.to_csv(raw_path, index=False)
        self.logger.info(f"原始数据已保存至 {raw_path}，样本数: {len(raw_df)}")
        
        # 如果需要生成模型兼容特征
        if generate_model_features:
            model_df = self._convert_raw_to_model_features(raw_samples)
            
            # 保存模型兼容特征数据
            model_path = os.path.join(output_dir, "model_features_data.csv")
            model_df.to_csv(model_path, index=False)
            self.logger.info(f"模型兼容特征数据已保存至 {model_path}，样本数: {len(model_df)}")
        
        # 分割训练集和测试集
        if split_train_test and test_size > 0:
            # 使用模型兼容特征进行分割（如果生成了的话，否则使用原始数据）
            df_to_split = model_df if generate_model_features else raw_df
            
            # 分割训练集和测试集
            train_df = df_to_split.sample(frac=1-test_size, random_state=42)
            test_df = df_to_split.drop(train_df.index)
            
            # 保存训练集
            train_path = os.path.join(output_dir, "train_data.csv")
            train_df.to_csv(train_path, index=False)
            self.logger.info(f"训练集已保存至 {train_path}，样本数: {len(train_df)}")
            
            # 保存测试集
            test_path = os.path.join(output_dir, "test_data.csv")
            test_df.to_csv(test_path, index=False)
            self.logger.info(f"测试集已保存至 {test_path}，样本数: {len(test_df)}")
            
            return {
                "train_path": train_path,
                "test_path": test_path,
                "train_count": len(train_df),
                "test_count": len(test_df)
            }
        else:
            # 保存完整数据集
            full_path = os.path.join(output_dir, "full_data.csv")
            df_to_save = model_df if generate_model_features else raw_df
            df_to_save.to_csv(full_path, index=False)
            self.logger.info(f"完整数据集已保存至 {full_path}，样本数: {len(df_to_save)}")
            
            return {
                "full_path": full_path,
                "count": len(df_to_save)
            }