"""
模型管理模块，负责异常检测模型的定义、创建、训练和选择

支持多种模型类型（传统机器学习、深度学习），并能根据协议类型选择最优模型
"""

from .base_model import BaseModel
from .traditional_models import (
    XGBoostModel,
    RandomForestModel,
    LogisticRegressionModel
)
from .deep_models import LSTMModel, MLPModel
from .model_factory import ModelFactory
from .model_selector import ModelSelector

__all__ = [
    "BaseModel",
    # 传统模型
    "XGBoostModel",
    "RandomForestModel",
    "LogisticRegressionModel",
    # 深度学习模型
    "LSTMModel",
    "MLPModel",
    # 模型管理
    "ModelFactory",
    "ModelSelector"
]
__version__ = "1.0.0"
