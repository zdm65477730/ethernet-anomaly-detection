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
            
            # 测试正常情况（修改模型返回值）
            self.mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])
            is_anomaly, score = self.model_detector.detect(self.normal_features)
            self.assertFalse(is_anomaly)
            self.assertAlmostEqual(score, 0.1)
    
    def test_rule_mode_detection(self):
        """测试纯规则模式的检测逻辑"""
        # 测试正常特征（不触发任何规则）
        is_anomaly, score = self.rule_detector.detect(self.normal_features)
        self.assertFalse(is_anomaly)
        self.assertEqual(score, 0.0)
        
        # 测试异常特征（应该触发规则）
        is_anomaly, score = self.rule_detector.detect(self.anomaly_features)
        self.assertTrue(is_anomaly)
        self.assertGreater(score, 0.0)  # 应该有一个正的异常分数
    
    def test_hybrid_mode_detection(self):
        """测试混合模式的检测逻辑"""
        with patch.object(self.hybrid_detector, 'model_factory', self.mock_factory):
            # 情况1：模型检测为异常，规则也检测为异常
            is_anomaly, score = self.hybrid_detector.detect(self.anomaly_features)
            self.assertTrue(is_anomaly)
            
            # 情况2：模型检测为异常，规则检测为正常
            self.mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
            is_anomaly, score = self.hybrid_detector.detect(self.normal_features)
            self.assertTrue(is_anomaly)
            
            # 情况3：模型检测为正常，规则检测为异常
            self.mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])
            is_anomaly, score = self.hybrid_detector.detect(self.anomaly_features)
            self.assertTrue(is_anomaly)
            
            # 情况4：两者都检测为正常
            is_anomaly, score = self.hybrid_detector.detect(self.normal_features)
            self.assertFalse(is_anomaly)
    
    def test_threshold_adjustment(self):
        """测试阈值调整对检测结果的影响"""
        # 创建不同阈值的检测器
        low_threshold_detector = AnomalyDetector(threshold=0.5, mode="model")
        high_threshold_detector = AnomalyDetector(threshold=0.9, mode="model")
        
        with patch.object(low_threshold_detector, 'model_factory', self.mock_factory), \
             patch.object(high_threshold_detector, 'model_factory', self.mock_factory):
            
            # 80%概率 > 50%阈值 → 异常
            is_anomaly, _ = low_threshold_detector.detect(self.normal_features)
            self.assertTrue(is_anomaly)
            
            # 80%概率 < 90%阈值 → 正常
            is_anomaly, _ = high_threshold_detector.detect(self.normal_features)
            self.assertFalse(is_anomaly)

class TestAlertManager(unittest.TestCase):
    """测试告警管理器"""
    
    def setUp(self):
        """初始化测试环境"""
        self.alert_manager = AlertManager()
        self.sample_features = {
            "packet_count": 1000,
            "avg_packet_size": 1500,
            "packet_rate": 100.0
        }
        self.session_id = "a1b2c3d4e5f6"
    
    @patch('src.detection.alert_manager.logging')
    def test_alert_triggering(self, mock_logging):
        """测试告警触发功能"""
        # 触发告警
        self.alert_manager.trigger_alert(
            features=self.sample_features,
            score=0.85,
            session_id=self.session_id,
            timestamp=1620000000.123456
        )
        
        # 验证日志被调用
        mock_logging.warning.assert_called_once()
        
        # 检查告警是否被记录
        self.assertEqual(len(self.alert_manager.alert_history), 1)
        alert = self.alert_manager.alert_history[0]
        self.assertEqual(alert['session_id'], self.session_id)
        self.assertEqual(alert['score'], 0.85)
        self.assertEqual(alert['level'], 'medium')  # 0.85属于中等风险
    
    @patch('src.detection.alert_manager.logging')
    def test_alert_cooldown(self, mock_logging):
        """测试告警冷却机制（防止告警风暴）"""
        # 第一次告警应该成功
        self.alert_manager.trigger_alert(
            features=self.sample_features,
            score=0.85,
            session_id=self.session_id,
            timestamp=1620000000.123456
        )
        self.assertEqual(len(self.alert_manager.alert_history), 1)
        
        # 立即触发第二次相同会话的告警，应该被冷却机制阻止
        self.alert_manager.trigger_alert(
            features=self.sample_features,
            score=0.85,
            session_id=self.session_id,
            timestamp=1620000001.123456  # 仅间隔1秒
        )
        self.assertEqual(len(self.alert_manager.alert_history), 1)  # 数量不变
        
        # 模拟时间过了冷却期（默认300秒）
        self.alert_manager.trigger_alert(
            features=self.sample_features,
            score=0.85,
            session_id=self.session_id,
            timestamp=1620000301.123456  # 间隔301秒
        )
        self.assertEqual(len(self.alert_manager.alert_history), 2)  # 应该新增一个

if __name__ == '__main__':
    unittest.main()
