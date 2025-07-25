import unittest
import numpy as np
from src.features.stat_extractor import StatFeatureExtractor
from src.features.temporal_extractor import TemporalFeatureExtractor

class TestStatFeatureExtractor(unittest.TestCase):
    """测试统计特征提取器"""
    
    def setUp(self):
        """初始化测试环境"""
        self.extractor = StatFeatureExtractor()
        # 创建一个模拟会话
        self.sample_session = {
            'packet_count': 10,
            'total_bytes': 1500,
            'packets': [
                {'length': 100, 'protocol': 6, 'tcp_flags': {'SYN': True, 'ACK': False}},
                {'length': 150, 'protocol': 6, 'tcp_flags': {'SYN': False, 'ACK': True}},
                {'length': 200, 'protocol': 6, 'tcp_flags': {'SYN': False, 'ACK': True}},
                {'length': 180, 'protocol': 6, 'tcp_flags': {'SYN': False, 'ACK': True}},
                {'length': 120, 'protocol': 6, 'tcp_flags': {'SYN': False, 'ACK': True}},
                {'length': 90, 'protocol': 6, 'tcp_flags': {'SYN': False, 'ACK': True}},
                {'length': 210, 'protocol': 6, 'tcp_flags': {'SYN': False, 'ACK': True}},
                {'length': 170, 'protocol': 6, 'tcp_flags': {'SYN': False, 'ACK': True}},
                {'length': 160, 'protocol': 6, 'tcp_flags': {'SYN': False, 'ACK': True}},
                {'length': 120, 'protocol': 6, 'tcp_flags': {'FIN': True}}
            ],
            'start_time': 1620000000.0,
            'end_time': 1620000010.0,
            'src_ip': '192.168.1.100',
            'dst_ip': '8.8.8.8',
            'protocol': 6  # TCP
        }
    
    def test_basic_stat_features(self):
        """测试基本统计特征提取"""
        features = self.extractor.extract_features_from_session(self.sample_session)
        
        # 验证基本统计特征
        self.assertEqual(features['packet_count'], 10)
        self.assertEqual(features['total_bytes'], 1500)
        self.assertEqual(features['avg_packet_size'], 150.0)
        self.assertAlmostEqual(features['std_packet_size'], 42.72, places=2)
        self.assertEqual(features['min_packet_size'], 90)
        self.assertEqual(features['max_packet_size'], 210)
    
    def test_tcp_specific_features(self):
        """测试TCP特定特征提取"""
        features = self.extractor.extract_features_from_session(self.sample_session)
        
        # 验证TCP特征
        self.assertEqual(features['tcp_syn_count'], 1)
        self.assertEqual(features['tcp_ack_count'], 8)
        self.assertEqual(features['tcp_fin_count'], 1)
        self.assertEqual(features['tcp_flag_ratio'], 1.0)  # 所有包都有标志位
    
    def test_duration_based_features(self):
        """测试基于持续时间的特征提取"""
        features = self.extractor.extract_features_from_session(self.sample_session)
        
        # 验证时间相关特征
        self.assertEqual(features['duration'], 10.0)
        self.assertEqual(features['packet_rate'], 1.0)  # 10包/10秒
        self.assertEqual(features['byte_rate'], 150.0)  # 1500字节/10秒

class TestTemporalFeatureExtractor(unittest.TestCase):
    """测试时序特征提取器"""
    
    def setUp(self):
        """初始化测试环境"""
        self.extractor = TemporalFeatureExtractor(window_sizes=[5, 10])
        # 创建一个带时间戳的模拟会话
        self.sample_session = {
            'packet_count': 5,
            'total_bytes': 750,
            'packets': [
                {'length': 100, 'timestamp': 1620000000.0},
                {'length': 150, 'timestamp': 1620000002.0},
                {'length': 200, 'timestamp': 1620000004.0},
                {'length': 180, 'timestamp': 1620000007.0},
                {'length': 120, 'timestamp': 1620000010.0}
            ],
            'start_time': 1620000000.0,
            'end_time': 1620000010.0,
            'protocol': 6  # TCP
        }
    
    def test_window_based_features(self):
        """测试基于时间窗口的特征提取"""
        features = self.extractor.extract_features_from_session(self.sample_session)
        
        # 验证5秒窗口特征（0-5秒有3个包，5-10秒有2个包）
        self.assertEqual(features['packet_rate_5s'], 0.6)  # 3包/5秒
        self.assertEqual(features['byte_rate_5s'], 90.0)   # 450字节/5秒
        
        # 验证10秒窗口特征
        self.assertEqual(features['packet_rate_10s'], 0.5)  # 5包/10秒
        self.assertEqual(features['byte_rate_10s'], 75.0)   # 750字节/10秒
    
    def test_inter_arrival_features(self):
        """测试包到达间隔特征提取"""
        features = self.extractor.extract_features_from_session(self.sample_session)
        
        # 包间隔分别是2, 2, 3, 3秒
        self.assertAlmostEqual(features['inter_arrival_mean'], 2.5, places=1)
        self.assertAlmostEqual(features['inter_arrival_std'], 0.5, places=1)
        self.assertEqual(features['inter_arrival_min'], 2.0)
        self.assertEqual(features['inter_arrival_max'], 3.0)
    
    def test_burst_detection(self):
        """测试流量突发检测特征"""
        # 创建一个有突发流量的会话
        burst_session = {
            'packet_count': 6,
            'packets': [
                {'length': 100, 'timestamp': 1620000000.0},
                {'length': 100, 'timestamp': 1620000000.1},  # 突发
                {'length': 100, 'timestamp': 1620000000.2},  # 突发
                {'length': 100, 'timestamp': 1620000001.0},
                {'length': 100, 'timestamp': 1620000002.0},
                {'length': 100, 'timestamp': 1620000003.0}
            ],
            'start_time': 1620000000.0,
            'end_time': 1620000003.0,
            'protocol': 6
        }
        
        features = self.extractor.extract_features_from_session(burst_session)
        
        # 应该检测到1次突发
        self.assertEqual(features['burst_count'], 1)
        self.assertEqual(features['burst_ratio'], 0.5)  # 3/6包在突发中

if __name__ == '__main__':
    unittest.main()
