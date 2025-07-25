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
            6: ("tcp", 0.6),   # TCP 占60%
            17: ("udp", 0.3),  # UDP 占30%
            1: ("icmp", 0.1)   # ICMP 占10%
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
        protocols, probabilities = zip(*self.protocols.items())
        return random.choices(protocols, weights=probabilities)[0]
    
    def _random_anomaly_type(self):
        """根据概率随机选择一个异常类型"""
        types, probabilities = zip(*self.anomaly_types.items())
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
        
        # 构建样本字典
        sample = {
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
        
        return sample
    
    def generate(self, num_samples=10000, output_dir="data/raw", split_train_test=True, 
                 test_size=0.2, start_time=None, force=False):
        """
        生成指定数量的模拟流量数据
        
        参数:
            num_samples: 样本数量
            output_dir: 输出目录
            split_train_test: 是否分割训练集和测试集
            test_size: 测试集比例
            start_time: 开始时间
            force: 如果目录已存在，是否强制覆盖
        """
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        else:
            if not force:
                self.logger.warning(f"目录 {output_dir} 已存在，使用force=True强制覆盖")
                return False
        
        self.logger.info(f"开始生成 {num_samples} 个模拟流量样本...")
        
        # 生成时间戳
        timestamps = self._generate_timestamps(num_samples, start_time)
        
        # 生成样本
        samples = []
        for i, ts in enumerate(timestamps):
            if i % 1000 == 0 and i > 0:
                self.logger.info(f"已生成 {i}/{num_samples} 个样本")
            
            sample = self.generate_sample(ts)
            samples.append(sample)
        
        # 转换为DataFrame
        df = pd.DataFrame(samples)
        
        # 显示数据分布
        self.logger.info(f"数据分布:")
        self.logger.info(f"  正常样本: {len(df[df['label'] == 0])} ({len(df[df['label'] == 0])/len(df):.2%})")
        self.logger.info(f"  异常样本: {len(df[df['label'] == 1])} ({len(df[df['label'] == 1])/len(df):.2%})")
        self.logger.info(f"  协议分布: \n{df['protocol_name'].value_counts().to_string()}")
        
        # 保存数据
        if split_train_test and test_size > 0:
            # 分割训练集和测试集
            train_df = df.sample(frac=1-test_size, random_state=42)
            test_df = df.drop(train_df.index)
            
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
            df.to_csv(full_path, index=False)
            self.logger.info(f"完整数据集已保存至 {full_path}，样本数: {len(df)}")
            
            return {
                "full_path": full_path,
                "count": len(df)
            }
    