import unittest
import time
from unittest.mock import Mock, patch
import numpy as np
from src.detection.anomaly_detector import AnomalyDetector
from src.detection.alert_manager import AlertManager
from src.models.model_factory import ModelFactory

class TestAnomalyDetector(unittest.TestCase):
    """测试异常检测逻辑"""
    
    def setUp(self):
        """初始化测试环境"""
        # 创建模拟模型
        self.mock_model = Mock()
        self.mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])  # 80%异常概率
        
        # 创建模型工厂的mock
        self.mock_factory = Mock(spec=ModelFactory)
        self.mock_factory.load_latest_model.return_value = self.mock_model
        
        # 初始化检测器（不同模式）
        self.model_detector = AnomalyDetector(threshold=0.7, mode="model")
        self.rule_detector = AnomalyDetector(threshold=0.7, mode="rule")
        self.hybrid_detector = AnomalyDetector(threshold=0.7, mode="hybrid")
        
        # 设置模拟特征
        self.normal_features = {
            "packet_count": 10,
            "avg_packet_size": 150,
            "packet_rate": 1.0,
            "tcp_syn_count": 1,
            "inter_arrival_std": 0.5
        }
        
        self.anomaly_features = {
            "packet_count": 1000,  # 异常多的包数量
            "avg_packet_size": 1500,  # 异常大的包
            "packet_rate": 100.0,  # 异常高的速率
            "tcp_syn_count": 50,  # 异常多的SYN包（可能是扫描）
            "inter_arrival_std": 5.0  # 异常不稳定的到达间隔
        }
    
    def test_model_mode_detection(self):
        """测试纯模型模式的检测逻辑"""
        # 使用模型工厂的mock
        with patch.object(self.model_detector, 'model_factory', self.mock_factory):
            # 测试异常情况（模型返回80%概率 > 70%阈值）
            is_anomaly, score = self.model_detector.detect(self.normal_features)
            self.assertTrue(is_anomaly)
            self.assertAlmostEqual(score, 0.8)
            
            # 测试正常情况（模型返回20%概率 < 70%阈值）
            self.mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
            is_anomaly, score = self.model_detector.detect(self.normal_features)
            self.assertFalse(is_anomaly)
            self.assertAlmostEqual(score, 0.2)
    
    def test_rule_mode_detection(self):
        """测试纯规则模式的检测逻辑"""
        # 测试正常流量
        is_anomaly, score = self.rule_detector.detect(self.normal_features)
        self.assertFalse(is_anomaly)
        self.assertLess(score, 0.7)  # 应该低于阈值
        
        # 测试异常流量
        is_anomaly, score = self.rule_detector.detect(self.anomaly_features)
        self.assertTrue(is_anomaly)
        self.assertGreaterEqual(score, 0.7)  # 应该达到或超过阈值
    
    def test_hybrid_mode_detection(self):
        """测试混合模式的检测逻辑"""
        # 使用模型工厂的mock
        with patch.object(self.hybrid_detector, 'model_factory', self.mock_factory):
            # 测试正常流量（规则和模型都正常）
            self.mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])  # 模型认为正常
            is_anomaly, score = self.hybrid_detector.detect(self.normal_features)
            self.assertFalse(is_anomaly)
            
            # 测试异常流量（规则和模型都认为异常）
            self.mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])  # 模型认为异常
            is_anomaly, score = self.hybrid_detector.detect(self.anomaly_features)
            self.assertTrue(is_anomaly)
    
    def test_missing_features(self):
        """测试缺失特征的处理"""
        incomplete_features = {
            "packet_count": 10,
            "avg_packet_size": 150
            # 缺少其他特征
        }
        
        # 纯模型模式应该可以处理（模型可以处理缺失特征）
        with patch.object(self.model_detector, 'model_factory', self.mock_factory):
            is_anomaly, score = self.model_detector.detect(incomplete_features)
            self.assertTrue(is_anomaly)  # 仍使用mock模型的返回值
            
        # 规则模式可能无法处理（缺少关键特征）
        try:
            is_anomaly, score = self.rule_detector.detect(incomplete_features)
        except KeyError:
            # 预期会因为缺少特征而抛出KeyError
            pass

class TestAlertManager(unittest.TestCase):
    """测试告警管理器"""
    
    def setUp(self):
        """初始化测试环境"""
        self.alert_manager = AlertManager()
        
        self.test_features = {
            "packet_count": 1000,
            "avg_packet_size": 1500,
            "packet_rate": 100.0,
            "tcp_syn_count": 50,
            "inter_arrival_std": 5.0
        }
        
        self.session_id = "test_session_123"
        self.anomaly_score = 0.85
    
    def test_alert_generation(self):
        """测试告警生成"""
        # 触发告警
        alert = self.alert_manager.trigger_alert(
            features=self.test_features,
            score=self.anomaly_score,
            session_id=self.session_id
        )
        
        # 验证告警信息
        self.assertIsNotNone(alert)
        self.assertEqual(alert['session_id'], self.session_id)
        self.assertEqual(alert['score'], self.anomaly_score)
        self.assertIn('timestamp', alert)
        self.assertIn('features', alert)
    
    def test_alert_storage(self):
        """测试告警存储"""
        # 触发多个告警
        alert1 = self.alert_manager.trigger_alert(self.test_features, 0.8, "session_1")
        alert2 = self.alert_manager.trigger_alert(self.test_features, 0.9, "session_2")
        
        # 验证告警存储
        alerts = self.alert_manager.get_recent_alerts(limit=10)
        self.assertGreaterEqual(len(alerts), 2)
        
        # 验证告警内容
        alert_ids = [alert['alert_id'] for alert in alerts]
        self.assertIn(alert1['alert_id'], alert_ids)
        self.assertIn(alert2['alert_id'], alert_ids)

if __name__ == '__main__':
    unittest.main()