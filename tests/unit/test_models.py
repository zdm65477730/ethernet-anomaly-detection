import unittest
import numpy as np
import pandas as pd
import tempfile  # 添加tempfile导入
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.base_model import BaseModel
from src.models.traditional_models import XGBoostModel, RandomForestModel
from src.models.deep_models import LSTMModel
from src.models.model_factory import ModelFactory
from src.models.model_selector import ModelSelector

class TestBaseModel(unittest.TestCase):
    """测试基础模型类接口"""
    
    def test_base_model_interface(self):
        """验证基础模型接口是否正确定义"""
        # 尝试实例化抽象类应该失败
        with self.assertRaises(TypeError):
            BaseModel(model_type="test")
    
        # 创建一个具体实现类
        class TestModel(BaseModel):
            def __init__(self):
                super().__init__(model_type="test")
                
            def fit(self, X, y, **kwargs):
                pass
            def predict(self, X):
                return np.array([0])
            def predict_proba(self, X):
                return np.array([[0.5, 0.5]])
            def save(self, file_path):
                pass
            @classmethod
            def load(cls, file_path):
                return cls()
    
        model = TestModel()
        self.assertEqual(model.model_type, "test")

class TestTraditionalModels(unittest.TestCase):
    """测试传统机器学习模型"""
    
    def setUp(self):
        """准备测试数据"""
        # 创建简单的测试数据
        self.X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6],
            'feature2': [2, 4, 6, 8, 10, 12]
        })
        self.y = pd.Series([0, 0, 0, 1, 1, 1])
    
    def test_xgboost_model(self):
        """测试XGBoost模型"""
        model = XGBoostModel()
        self.assertEqual(model.model_type, "xgboost")
    
    def test_random_forest_model(self):
        """测试随机森林模型"""
        model = RandomForestModel()
        self.assertEqual(model.model_type, "random_forest")
    
    def test_model_persistence(self):
        """测试模型持久化"""
        model = XGBoostModel(n_estimators=10)
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            # 保存模型
            model.save(temp_path)
            # 检查文件是否存在
            self.assertTrue(os.path.exists(temp_path))
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)

class TestDeepModels(unittest.TestCase):
    """测试深度学习模型"""

    def test_lstm_model(self):
        """测试LSTM模型"""
        model = LSTMModel()
        self.assertEqual(model.model_type, "lstm")

class TestModelFactory(unittest.TestCase):
    """测试模型工厂类"""
    
    def test_create_model(self):
        """测试创建不同类型的模型"""
        factory = ModelFactory()
        
        # 测试创建XGBoost模型
        xgb_model = factory.create_model("xgboost", n_estimators=100)
        self.assertIsInstance(xgb_model, XGBoostModel)
        
        # 测试创建随机森林模型
        rf_model = factory.create_model("random_forest", n_estimators=50)
        self.assertIsInstance(rf_model, RandomForestModel)
        
        # 测试创建LSTM模型
        lstm_model = factory.create_model("lstm", input_dim=10, hidden_dim=32)
        self.assertIsInstance(lstm_model, LSTMModel)
        
        # 测试创建未知模型类型（应该抛出错误）
        with self.assertRaises(ValueError):
            factory.create_model("unknown_model")

class TestModelSelector(unittest.TestCase):
    """测试模型选择器"""
    
    def setUp(self):
        """初始化测试环境"""
        self.selector = ModelSelector()
        # 添加一些性能数据
        self.selector.update_performance("tcp", "xgboost", {"f1": 0.85, "precision": 0.82, "recall": 0.88})
        self.selector.update_performance("tcp", "lstm", {"f1": 0.89, "precision": 0.87, "recall": 0.91})
        self.selector.update_performance("udp", "xgboost", {"f1": 0.78, "precision": 0.76, "recall": 0.80})
        self.selector.update_performance("udp", "random_forest", {"f1": 0.82, "precision": 0.80, "recall": 0.84})
    
    def test_select_best_model(self):
        """测试为特定协议选择最佳模型"""
        # TCP协议应该选择LSTM（F1更高）
        best_tcp = self.selector.select_best_model("tcp")
        self.assertEqual(best_tcp, "lstm")
        
        # UDP协议应该选择随机森林（F1更高）
        best_udp = self.selector.select_best_model("udp")
        self.assertEqual(best_udp, "random_forest")
        
        # 测试使用协议编号（6是TCP）
        best_tcp_by_num = self.selector.select_best_model(6)
        self.assertEqual(best_tcp_by_num, "lstm")
        
        # 对未知协议应该返回默认模型
        best_unknown = self.selector.select_best_model("icmp")
        self.assertEqual(best_unknown, "xgboost")  # 默认模型
        
    def test_get_best_model_no_data(self):
        """测试没有性能数据时获取最佳模型"""
        # 创建一个新的没有性能数据的selector
        empty_selector = ModelSelector()
        
        # 当没有数据时应该返回默认模型
        best_model = empty_selector.select_best_model("tcp")
        self.assertEqual(best_model, "xgboost")
        
        # 测试多种未知协议
        protocols = ["icmp", "arp", "http", 17]  # 包含字符串和数字
        for proto in protocols:
            best_model = empty_selector.select_best_model(proto)
            self.assertEqual(best_model, "xgboost")
            
    def test_get_supported_protocols(self):
        """测试获取支持的协议列表"""
        protocols = self.selector.get_supported_protocols()
        self.assertEqual(set(protocols), set(["tcp", "udp"]))
        
        # 新建的selector应该返回空列表
        empty_selector = ModelSelector()
        self.assertEqual(empty_selector.get_supported_protocols(), [])

if __name__ == '__main__':
    unittest.main()