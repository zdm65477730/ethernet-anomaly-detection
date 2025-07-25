import abc
import time
from typing import Optional, Dict, Union
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

class BaseModel(metaclass=abc.ABCMeta):
    """
    模型基类，定义所有模型必须实现的接口
    """
    def __init__(self, model_type: str, **kwargs):
        self.model_type = model_type  # 模型类型（如"xgboost"）
        self.model = None  # 实际模型对象
        self.params = kwargs  # 模型参数
        self.logger = get_logger(f"model.{model_type}")
        self.is_trained = False  # 训练状态
        self.feature_names = None  # 特征名称列表
        self.train_timestamp = None  # 训练时间戳
        self.metrics = {}  # 训练后的评估指标

    @abc.abstractmethod
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],** kwargs
    ) -> None:
        """
        全量训练模型

        参数:
            X: 特征数据
            y: 标签数据（0=正常，1=异常）
            **kwargs: 训练参数（如epochs、batch_size等）
        """
        self.feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        self.train_timestamp = time.time()

    @abc.abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        预测标签（0/1）

        参数:
            X: 特征数据

        返回:
            预测标签数组
        """
        if not self.is_trained:
            self.logger.warning("模型未训练，返回默认预测")
            return np.zeros(len(X))

    @abc.abstractmethod
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        预测异常概率

        参数:
            X: 特征数据

        返回:
            异常概率数组（形状为(n_samples, 2)，[:,1]为异常概率）
        """
        if not self.is_trained:
            self.logger.warning("模型未训练，返回默认概率")
            return np.zeros((len(X), 2))

    def partial_fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],** kwargs
    ) -> None:
        """
        增量训练（可选实现）

        参数:
            X: 新的特征数据
            y: 新的标签数据
            **kwargs: 训练参数
        """
        raise NotImplementedError(f"{self.model_type} 不支持增量训练")

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        获取特征重要性（可选实现）

        返回:
            特征名称到重要性的字典
        """
        self.logger.warning(f"{self.model_type} 不支持特征重要性")
        return None

    @abc.abstractmethod
    def save(self, file_path: str) -> None:
        """
        保存模型到文件
        """
        raise NotImplementedError("子类必须实现save方法")

    @classmethod
    @abc.abstractmethod
    def load(cls, file_path: str) -> "BaseModel":
        """
        从文件加载模型
        """
        raise NotImplementedError("子类必须实现load方法")

    def get_metadata(self) -> Dict[str, any]:
        """返回模型元数据"""
        return {
            "model_type": self.model_type,
            "is_trained": self.is_trained,
            "params": self.params,
            "feature_names": self.feature_names,
            "train_timestamp": self.train_timestamp,
            "metrics": self.metrics
        }

    def __str__(self) -> str:
        return f"{self.model_type}(trained={self.is_trained}, params={self.params})"
