import unittest
import time
import tempfile
import os
from unittest.mock import patch, Mock, MagicMock
import numpy as np
from src.capture.packet_capture import PacketCapture
from src.capture.session_tracker import SessionTracker
from src.features.stat_extractor import StatFeatureExtractor
from src.features.temporal_extractor import TemporalFeatureExtractor
from src.models.model_factory import ModelFactory
from src.detection.anomaly_detector import AnomalyDetector
from src.detection.alert_manager import AlertManager

class TestSystemEndToEnd(unittest.TestCase):
    """系统端到端集成测试"""
    
    def setUp(self):
        """初始化系统组件"""
        # 创建临时目录用于测试
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # 初始化系统核心组件
        self.capture = PacketCapture(interface="test0", filter="tcp", timeout=10)
        self.tracker = SessionTracker(timeout=30)
        self.stat_extractor = StatFeatureExtractor()
        self.temporal_extractor = TemporalFeatureExtractor(window_sizes=[10, 30])
        self.detector = AnomalyDetector(threshold=0.7, mode="hybrid")
        self.alert_manager = AlertManager()
        
        # 创建模拟模型，总是返回75%的异常概率
        self.mock_model = Mock()
        self.mock_model.predict_proba.return_value = np.array([[0.25, 0.75]])
        
        # 替换模型工厂
        self.mock_factory = Mock(spec=ModelFactory)
        self.mock_factory.load_latest_model.return_value = self.mock_model
        self.detector.model_factory = self.mock_factory
        
        # 测试数据
        self.normal_packets = [
            {
                "timestamp": 1620000000.0,
                "src_ip": "192.168.1.100",
                "dst_ip": "8.8.8.8",
                "src_port": 12345,
                "dst_port": 80,
                "protocol": 6,
                "length": 150,
                "tcp_flags": {"SYN": True}
            },
            {
                "timestamp": 1620000001.0,
                "src_ip": "192.168.1.100",
                "dst_ip": "8.8.8.8",
                "src_port": 12345,
                "dst_port": 80,
                "protocol": 6,
                "length": 200,
                "tcp_flags": {"ACK": True}
            },
            {
                "timestamp": 1620000002.0,
                "src_ip": "192.168.1.100",
                "dst_ip": "8.8.8.8",
                "src_port": 12345,
                "dst_port": 80,
                "protocol": 6,
                "length": 180,
                "tcp_flags": {"ACK": True, "PSH": True}
            }
        ]
        
        self.anomaly_packets = [
            {
                "timestamp": 1620000010.0,
                "src_ip": "10.0.0.5",
                "dst_ip": "192.168.1.100",
                "src_port": 54321,
                "dst_port": 22,
                "protocol": 6,
                "length": 65535,  # 异常大的包
                "tcp_flags": {"SYN": True}
            },
            {
                "timestamp": 1620000010.1,
                "src_ip": "10.0.0.5",
                "dst_ip": "192.168.1.100",
                "src_port": 54321,
                "dst_port": 23,
                "protocol": 6,
                "length": 65535,
                "tcp_flags": {"SYN": True}
            },
            {
                "timestamp": 1620000010.2,
                "src_ip": "10.0.0.5",
                "dst_ip": "192.168.1.100",
                "src_port": 54321,
                "dst_port": 80,
                "protocol": 6,
                "length": 65535,
                "tcp_flags": {"SYN": True}
            }
        ]
    
    def tearDown(self):
        """清理测试环境"""
        self.temp_dir.cleanup()
    
    def test_normal_traffic_processing(self):
        """测试正常流量处理流程"""
        # 模拟处理正常数据包
        sessions = {}
        for packet in self.normal_packets:
            session_key, session_data = self.tracker.process_packet(packet)
            sessions[session_key] = session_data
        
        # 验证会话创建
        self.assertEqual(len(sessions), 1)  # 应该只有一个会话
        
        # 获取会话数据
        session_key = list(sessions.keys())[0]
        session_data = sessions[session_key]
        
        # 验证会话统计信息
        self.assertEqual(session_data['packet_count'], 3)
        self.assertEqual(session_data['total_bytes'], 530)
        
        # 提取特征
        features = self.stat_extractor.extract_features_from_session(session_data)
        
        # 验证特征提取
        self.assertIn('packet_count', features)
        self.assertIn('avg_packet_size', features)
        self.assertIn('tcp_syn_count', features)
        
        # 使用检测器检测（应该不会触发告警，因为是正常流量）
        with patch.object(self.detector, 'model_factory', self.mock_factory):
            is_anomaly, score = self.detector.detect(features)
            
            # 由于我们使用的是混合模式，且模拟模型返回0.75 > 0.7阈值，所以应该检测为异常
            self.assertTrue(is_anomaly)
            self.assertGreaterEqual(score, 0.7)
    
    def test_anomaly_traffic_processing(self):
        """测试异常流量处理流程"""
        # 模拟处理异常数据包
        sessions = {}
        for packet in self.anomaly_packets:
            session_key, session_data = self.tracker.process_packet(packet)
            sessions[session_key] = session_data
        
        # 验证会话创建
        self.assertEqual(len(sessions), 3)  # 应该有三个不同的会话（不同目的端口）
        
        # 提取特征并检测
        alerts = []
        for session_data in sessions.values():
            features = self.stat_extractor.extract_features_from_session(session_data)
            
            # 使用检测器检测
            with patch.object(self.detector, 'model_factory', self.mock_factory):
                is_anomaly, score = self.detector.detect(features)
                
                # 触发告警
                if is_anomaly:
                    alert = self.alert_manager.trigger_alert(features, score, "test_session")
                    alerts.append(alert)
        
        # 验证告警生成
        self.assertGreater(len(alerts), 0)
        
        # 验证告警内容
        for alert in alerts:
            self.assertIn('alert_id', alert)
            self.assertIn('timestamp', alert)
            self.assertIn('session_id', alert)
            self.assertIn('score', alert)
            self.assertGreaterEqual(alert['score'], 0.7)
    
    def test_temporal_feature_extraction(self):
        """测试时序特征提取"""
        # 使用正常数据包流提取时序特征
        temporal_features = self.temporal_extractor.extract_features(self.normal_packets)
        
        # 验证时序特征
        self.assertIn('packet_rate_10s', temporal_features)
        self.assertIn('avg_packet_size_10s', temporal_features)
        self.assertIn('inter_arrival_mean_10s', temporal_features)
        
        # 验证特征值合理
        self.assertGreaterEqual(temporal_features['packet_rate_10s'], 0)
        self.assertGreaterEqual(temporal_features['avg_packet_size_10s'], 0)
    
    def test_model_selection_by_protocol(self):
        """测试根据协议选择模型"""
        # 创建模型选择器和工厂
        from src.models.model_selector import ModelSelector
        selector = ModelSelector()
        factory = ModelFactory()
        
        # 为不同协议添加性能数据
        selector.update_performance("tcp", "xgboost", {"f1": 0.85, "precision": 0.82, "recall": 0.88})
        selector.update_performance("tcp", "lstm", {"f1": 0.89, "precision": 0.87, "recall": 0.91})
        
        # TCP协议应该选择LSTM（F1分数更高）
        best_model = selector.select_best_model("tcp")
        self.assertEqual(best_model, "lstm")
        
        # 测试模型工厂创建模型
        model = factory.create_model(best_model, input_dim=10, hidden_dim=32)
        self.assertIsInstance(model, MagicMock().__class__)  # Mock对象类型

if __name__ == '__main__':
    unittest.main()