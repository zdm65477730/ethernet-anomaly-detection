import unittest
import os
import tempfile
import numpy as np
from unittest.mock import patch, Mock
from src.models.base_model import BaseModel
from src.models.model_factory import ModelFactory
from src.models.traditional_models import XGBoostModel, RandomForestModel
from src.models.deep_models import LSTMModel
from src.models.model_selector import ModelSelector

class TestBaseModel(unittest.TestCase):
    """测试基础模型类接口"""
    
    def test_base_model_interface(self):
        """验证基础模型接口是否正确定义"""
        # 尝试实例化抽象类应该失败
        with self.assertRaises(TypeError):
            BaseModel()
        
        # 创建一个具体实现类
        class TestModel(BaseModel):
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
        self.assertTrue(hasattr(model, 'fit'))
        self.assertTrue(hasattr(model, 'predict'))
        self.assertTrue(hasattr(model, 'predict_proba'))
        self.assertTrue(hasattr(model, 'save'))
        self.assertTrue(hasattr(TestModel, 'load'))

class TestTraditionalModels(unittest.TestCase):
    """测试传统机器学习模型"""
    
    def setUp(self):
        """创建测试数据"""
        # 创建简单的二分类测试数据
        self.X = np.array([
            [1.2, 3.4, 2.1],
            [0.8, 2.9, 1.7],
            [5.6, 4.1, 3.2],
            [6.2, 3.8, 4.5],
            [2.1, 2.2, 1.9],
            [5.9, 4.0, 3.8]
        ])
        self.y = np.array([0, 0, 1, 1, 0, 1])  # 0=正常, 1=异常
    
    def test_xgboost_model(self):
        """测试XGBoost模型"""
        model = XGBoostModel(n_estimators=10, max_depth=3)
        
        # 测试训练
        model.fit(self.X, self.y)
        
        # 测试预测
        predictions = model.predict(self.X)
        self.assertEqual(predictions.shape, (6,))
        self.assertTrue(np.all((predictions == 0) | (predictions == 1)))
        
        # 测试概率预测
        probabilities = model.predict_proba(self.X)
        self.assertEqual(probabilities.shape, (6, 2))
        self.assertTrue(np.allclose(probabilities.sum(axis=1), np.ones(6)))
    
    def test_random_forest_model(self):
        """测试随机森林模型"""
        model = RandomForestModel(n_estimators=10, max_depth=3)
        
        # 测试训练
        model.fit(self.X, self.y)
        
        # 测试预测
        predictions = model.predict(self.X)
        self.assertEqual(predictions.shape, (6,))
        self.assertTrue(np.all((predictions == 0) | (predictions == 1)))
        
        # 测试概率预测
        probabilities = model.predict_proba(self.X)
        self.assertEqual(probabilities.shape, (6, 2))
        self.assertTrue(np.allclose(probabilities.sum(axis=1), np.ones(6)))
    
    def test_model_persistence(self):
        """测试模型保存和加载功能"""
        model = XGBoostModel(n_estimators=10)
        model.fit(self.X, self.y)
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            # 保存模型
            model.save(temp_path)
            
            # 加载模型
            loaded_model = XGBoostModel.load(temp_path)
            
            # 验证预测结果一致
            original_preds = model.predict(self.X)
            loaded_preds = loaded_model.predict(self.X)
            np.testing.assert_array_equal(original_preds, loaded_preds)
        finally:
            os.unlink(temp_path)

class TestDeepModels(unittest.TestCase):
    """测试深度学习模型"""
    
    def setUp(self):
        """创建测试数据"""
        # LSTM需要3D输入 (samples, timesteps, features)
        self.X = np.random.rand(10, 5, 3)  # 10个样本，5个时间步，3个特征
        self.y = np.random.randint(0, 2, size=10)  # 二分类标签
    
    def test_lstm_model(self):
        """测试LSTM模型"""
        model = LSTMModel(input_dim=3, hidden_dim=16, num_layers=1)
        
        # 测试训练
        model.fit(self.X, self.y, epochs=5, batch_size=4)
        
        # 测试预测
        predictions = model.predict(self.X)
        self.assertEqual(predictions.shape, (10,))
        self.assertTrue(np.all((predictions == 0) | (predictions == 1)))
        
        # 测试概率预测
        probabilities = model.predict_proba(self.X)
        self.assertEqual(probabilities.shape, (10, 2))
        self.assertTrue(np.allclose(probabilities.sum(axis=1), np.ones(10), atol=0.01))

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

if __name__ == '__main__':
    unittest.main()
