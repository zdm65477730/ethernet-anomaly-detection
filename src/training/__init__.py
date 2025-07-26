"""
模型训练模块，负责模型的基础训练、增量训练、持续训练和评估

提供完整的模型训练流水线，支持传统机器学习和深度学习模型
"""

from .continuous_trainer import ContinuousTrainer
from .incremental_trainer import IncrementalTrainer
from .model_evaluator import ModelEvaluator
from .model_trainer import ModelTrainer
from .feedback_optimizer import FeedbackOptimizer
from .automl_trainer import AutoMLTrainer

__all__ = [
    "ContinuousTrainer",
    "IncrementalTrainer",
    "ModelEvaluator",
    "ModelTrainer",
    "FeedbackOptimizer",
    "AutoMLTrainer"
]
__version__ = "1.0.0"