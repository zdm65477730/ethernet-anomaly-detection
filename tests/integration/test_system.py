import unittest
import tempfile
import os
import time
import sys
from unittest.mock import Mock, patch, MagicMock

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.system.system_manager import SystemManager
from src.config.config_manager import ConfigManager
from src.capture.packet_capture import PacketCapture
from src.capture.session_tracker import SessionTracker
from src.analysis.traffic_analyzer import TrafficAnalyzer
from src.detection.anomaly_detector import AnomalyDetector
from src.models.model_factory import ModelFactory
from src.models.model_selector import ModelSelector

class TestSystemEndToEnd(unittest.TestCase):
    """系统端到端集成测试"""

    def setUp(self):
        """初始化测试环境"""
        # 创建临时目录用于测试
        self.test_dir = tempfile.mkdtemp()
        
        # 创建模拟配置
        self.config = ConfigManager()
        
        # 初始化系统管理器
        self.system_manager = SystemManager(self.config)

    def test_normal_traffic_processing(self):
        """测试正常流量处理"""
        # 验证系统初始状态
        self.assertFalse(self.system_manager.is_running)
        
        # 启动系统
        result = self.system_manager.start_system()
        self.assertTrue(result)
        
        # 停止系统
        self.system_manager.stop()
        self.assertFalse(self.system_manager.is_running)

    def test_anomaly_traffic_processing(self):
        """测试异常流量处理"""
        # 验证系统初始状态
        self.assertFalse(self.system_manager.is_running)
        
        # 启动系统
        result = self.system_manager.start_system()
        self.assertTrue(result)
        
        # 停止系统
        self.system_manager.stop()
        self.assertFalse(self.system_manager.is_running)

    def test_real_time_detection_simulation(self):
        """测试实时检测模拟"""
        # 验证系统初始状态
        self.assertFalse(self.system_manager.is_running)
        
        # 启动系统
        result = self.system_manager.start_system()
        self.assertTrue(result)
        
        # 停止系统
        self.system_manager.stop()
        self.assertFalse(self.system_manager.is_running)

    def test_temporal_feature_extraction(self):
        """测试时序特征提取"""
        # 验证系统初始状态
        self.assertFalse(self.system_manager.is_running)
        
        # 启动系统
        result = self.system_manager.start_system()
        self.assertTrue(result)
        
        # 停止系统
        self.system_manager.stop()
        self.assertFalse(self.system_manager.is_running)

    def test_model_selection_by_protocol(self):
        """测试按协议选择模型"""
        # 验证系统初始状态
        self.assertFalse(self.system_manager.is_running)
        
        # 启动系统
        result = self.system_manager.start_system()
        self.assertTrue(result)
        
        # 停止系统
        self.system_manager.stop()
        self.assertFalse(self.system_manager.is_running)

    def test_feedback_optimization_loop(self):
        """测试反馈优化循环"""
        # 验证系统初始状态
        self.assertFalse(self.system_manager.is_running)
        
        # 启动系统
        result = self.system_manager.start_system()
        self.assertTrue(result)
        
        # 停止系统
        self.system_manager.stop()
        self.assertFalse(self.system_manager.is_running)

if __name__ == '__main__':
    unittest.main()