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
        if self.capture.is_running:
            self.capture.stop()
    
    def test_normal_traffic_flow(self):
        """测试正常流量处理流程"""
        # 1. 处理正常数据包
        session_ids = set()
        for packet in self.normal_packets:
            session_id, session = self.tracker.track_packet(packet)
            session_ids.add(session_id)
        
        # 应该只有一个会话
        self.assertEqual(len(session_ids), 1)
        session_id = next(iter(session_ids))
        session = self.tracker.get_session(session_id)
        
        # 2. 提取特征
        stat_features = self.stat_extractor.extract_features_from_session(session)
        temporal_features = self.temporal_extractor.extract_features_from_session(session)
        all_features = {** stat_features, **temporal_features}
        
        # 验证特征数量
        self.assertGreater(len(all_features), 0)
        
        # 3. 异常检测
        is_anomaly, score = self.detector.detect(all_features)
        
        # 正常流量在模型返回75%概率下应该被检测为异常
        # 但根据我们的规则，正常流量不应该触发规则检测
        # 所以混合模式下会被检测为异常（因为模型检测为异常）
        self.assertTrue(is_anomaly)
        self.assertAlmostEqual(score, 0.75)
        
        # 4. 触发告警
        self.alert_manager.trigger_alert(all_features, score, session_id, time.time())
        self.assertEqual(len(self.alert_manager.alert_history), 1)
    
    def test_anomaly_traffic_flow(self):
        """测试异常流量处理流程"""
        # 1. 处理异常数据包
        session_ids = set()
        for packet in self.anomaly_packets:
            session_id, session = self.tracker.track_packet(packet)
            session_ids.add(session_id)
        
        # 异常扫描应该创建多个会话（每个端口一个）
        self.assertEqual(len(session_ids), 3)
        
        # 2. 对第一个异常会话进行处理
        session_id = next(iter(session_ids))
        session = self.tracker.get_session(session_id)
        
        # 3. 提取特征
        stat_features = self.stat_extractor.extract_features_from_session(session)
        temporal_features = self.temporal_extractor.extract_features_from_session(session)
        all_features = {** stat_features, **temporal_features}
        
        # 4. 异常检测（混合模式）
        is_anomaly, score = self.detector.detect(all_features)
        
        # 异常流量应该被检测出来
        self.assertTrue(is_anomaly)
        self.assertGreater(score, 0.75)  # 应该高于模型单独给出的分数
        
        # 5. 触发告警
        self.alert_manager.trigger_alert(all_features, score, session_id, time.time())
        self.assertEqual(len(self.alert_manager.alert_history), 1)
    
    @patch('src.capture.packet_capture.PacketCapture.get_packet')
    def test_full_system_integration(self, mock_get_packet):
        """测试完整系统流程（从抓包到告警）"""
        # 模拟数据包捕获
        mock_get_packet.side_effect = [
            self.normal_packets[0],
            self.normal_packets[1],
            self.normal_packets[2],
            None  # 表示没有更多数据包
        ]
        
        # 启动抓包
        self.capture.start()
        time.sleep(0.5)  # 等待处理
        
        # 验证会话被创建
        self.assertEqual(len(self.tracker.sessions), 1)
        
        # 处理会话并检测异常
        session_id, session = next(iter(self.tracker.sessions.items()))
        stat_features = self.stat_extractor.extract_features_from_session(session)
        temporal_features = self.temporal_extractor.extract_features_from_session(session)
        all_features = {** stat_features, **temporal_features}
        
        is_anomaly, score = self.detector.detect(all_features)
        self.assertTrue(is_anomaly)
        
        # 触发告警
        self.alert_manager.trigger_alert(all_features, score, session_id, time.time())
        self.assertEqual(len(self.alert_manager.alert_history), 1)
        
        # 停止抓包
        self.capture.stop()

if __name__ == '__main__':
    unittest.main()
