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
        self.assertEqual(features['tcp_fin_count'], 1)
        self.assertEqual(features['tcp_packet_ratio'], 1.0)  # 所有包都是TCP包
        self.assertGreater(features['session_duration'], 0)
        self.assertGreater(features['bytes_per_second'], 0)
    
    def test_protocol_distribution(self):
        """测试协议分布特征"""
        # 创建包含多种协议的会话
        mixed_session = self.sample_session.copy()
        mixed_session['packets'] = [
            {'length': 100, 'protocol': 6, 'tcp_flags': {'SYN': True}},   # TCP
            {'length': 150, 'protocol': 17},  # UDP
            {'length': 200, 'protocol': 1},   # ICMP
            {'length': 180, 'protocol': 6},   # TCP
            {'length': 120, 'protocol': 17},  # UDP
        ]
        mixed_session['packet_count'] = 5
        mixed_session['total_bytes'] = 750
        
        features = self.extractor.extract_features_from_session(mixed_session)
        
        # 验证协议分布特征
        self.assertAlmostEqual(features['tcp_packet_ratio'], 0.4)  # 2/5
        self.assertAlmostEqual(features['udp_packet_ratio'], 0.4)  # 2/5
        self.assertAlmostEqual(features['icmp_packet_ratio'], 0.2) # 1/5
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 空会话
        empty_session = {
            'packet_count': 0,
            'total_bytes': 0,
            'packets': [],
            'start_time': 1620000000.0,
            'end_time': 1620000000.0,
            'src_ip': '192.168.1.100',
            'dst_ip': '8.8.8.8',
            'protocol': 6
        }
        
        features = self.extractor.extract_features_from_session(empty_session)
        self.assertEqual(features['packet_count'], 0)
        self.assertEqual(features['total_bytes'], 0)
        self.assertEqual(features['avg_packet_size'], 0)
        
        # 单包会话
        single_packet_session = {
            'packet_count': 1,
            'total_bytes': 100,
            'packets': [{'length': 100, 'protocol': 6, 'tcp_flags': {'SYN': True}}],
            'start_time': 1620000000.0,
            'end_time': 1620000001.0,
            'src_ip': '192.168.1.100',
            'dst_ip': '8.8.8.8',
            'protocol': 6
        }
        
        features = self.extractor.extract_features_from_session(single_packet_session)
        self.assertEqual(features['packet_count'], 1)
        self.assertEqual(features['total_bytes'], 100)
        self.assertEqual(features['avg_packet_size'], 100)
        self.assertEqual(features['std_packet_size'], 0)  # 单个值标准差为0

class TestTemporalFeatureExtractor(unittest.TestCase):
    """测试时序特征提取器"""
    
    def setUp(self):
        """初始化测试环境"""
        self.extractor = TemporalFeatureExtractor(window_sizes=[10, 30, 60])  # 增加一个典型的窗口大小
        # 创建模拟数据包流
        self.packet_stream = [
            {'timestamp': 1620000000.0, 'length': 100, 'protocol': 6},
            {'timestamp': 1620000001.0, 'length': 150, 'protocol': 6},
            {'timestamp': 1620000002.0, 'length': 200, 'protocol': 17},
            {'timestamp': 1620000003.0, 'length': 180, 'protocol': 6},
            {'timestamp': 1620000004.0, 'length': 120, 'protocol': 17},
            {'timestamp': 1620000005.0, 'length': 90, 'protocol': 1},
            {'timestamp': 1620000006.0, 'length': 210, 'protocol': 6},
            {'timestamp': 1620000007.0, 'length': 170, 'protocol': 6},
            {'timestamp': 1620000008.0, 'length': 160, 'protocol': 17},
            {'timestamp': 1620000009.0, 'length': 120, 'protocol': 6},
        ]
    
    def test_temporal_features(self):
        """测试时序特征提取"""
        features = self.extractor.extract_features(self.packet_stream)
        
        # 验证基本时序特征存在
        self.assertIn('packet_rate_10s', features)
        self.assertIn('avg_packet_size_10s', features)
        self.assertIn('protocol_diversity_10s', features)
        self.assertIn('inter_arrival_mean_10s', features)
        self.assertIn('inter_arrival_std_10s', features)
        
        # 验证特征值合理
        self.assertGreaterEqual(features['packet_rate_10s'], 0)
        self.assertGreaterEqual(features['avg_packet_size_10s'], 0)
        self.assertGreaterEqual(features['protocol_diversity_10s'], 0)
        self.assertGreaterEqual(features['inter_arrival_mean_10s'], 0)
    
    def test_multiple_window_sizes(self):
        """测试多时间窗口特征提取"""
        features = self.extractor.extract_features(self.packet_stream)
        
        # 验证两个时间窗口的特征都存在
        self.assertIn('packet_rate_10s', features)
        self.assertIn('packet_rate_30s', features)
        self.assertIn('avg_packet_size_10s', features)
        self.assertIn('avg_packet_size_30s', features)
    
    def test_empty_packet_stream(self):
        """测试空数据包流处理"""
        empty_stream = []
        features = self.extractor.extract_features(empty_stream)
        
        # 验证空流处理
        self.assertEqual(features['packet_rate_10s'], 0)
        self.assertEqual(features['avg_packet_size_10s'], 0)
        self.assertEqual(features['protocol_diversity_10s'], 0)

if __name__ == '__main__':
    unittest.main()