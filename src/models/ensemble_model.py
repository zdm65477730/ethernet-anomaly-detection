import os
import pickle
import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, List
import time
from src.utils.logger import get_logger
from .base_model import BaseModel
from .traditional_models import XGBoostModel, RandomForestModel
from .deep_models import MLPModel

class EnsembleModel(BaseModel):
    """集成模型，结合多种模型的优势"""
    
    def __init__(self, **kwargs):
        # 从kwargs中提取model_type，如果不存在则默认为"ensemble"
        model_type = kwargs.pop("model_type", "ensemble")
        
        default_params = {
            "models": ["xgboost", "random_forest", "mlp"],  # 集成的模型类型
            "weights": None,  # 模型权重，默认为等权重
            "voting": "soft",  # 投票方式：'hard' 或 'soft'
            "threshold": 0.5   # 分类阈值
        }
        params = {**default_params, **kwargs}
        super().__init__(model_type=model_type, **params)
        self.models = []  # 存储训练好的模型
        self.model_weights = None  # 模型权重
        self.logger = get_logger("model.ensemble")
        
        # 初始化模型权重
        if self.params["weights"] is None:
            # 等权重
            n_models = len(self.params["models"])
            self.model_weights = [1.0/n_models] * n_models
        else:
            self.model_weights = self.params["weights"]
        
        # 确保model_weights不为None
        if self.model_weights is None:
            n_models = len(self.params["models"])
            self.model_weights = [1.0/n_models] * n_models

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray], **kwargs
    ) -> None:
        """训练所有子模型"""
        super().fit(X, y, **kwargs)
        self.models = []
        
        self.logger.info(f"开始训练集成模型，包含 {len(self.params['models'])} 个子模型")
        
        # 训练每个子模型
        for i, model_type in enumerate(self.params["models"]):
            self.logger.info(f"训练 {model_type} 子模型...")
            
            # 根据模型类型创建并训练子模型
            if model_type == "xgboost":
                model = XGBoostModel(model_type="xgboost")
            elif model_type == "random_forest":
                model = RandomForestModel(model_type="random_forest")
            elif model_type == "mlp":
                model = MLPModel(model_type="mlp")
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            # 训练子模型
            model.fit(X, y, **kwargs)
            self.models.append(model)
            
            self.logger.info(f"{model_type} 子模型训练完成")
        
        self.is_trained = True
        self.logger.info("集成模型训练完成")

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """预测标签"""
        super().predict(X)
        
        if self.params["voting"] == "hard":
            # 硬投票
            predictions = []
            for i, model in enumerate(self.models):
                pred = model.predict(X)
                predictions.append(pred * self.model_weights[i])
            
            # 加权平均并应用阈值
            weighted_pred = np.mean(predictions, axis=0)
            return (weighted_pred >= self.params["threshold"]).astype(int)
        else:
            # 软投票
            probas = self.predict_proba(X)
            return (probas[:, 1] >= self.params["threshold"]).astype(int)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """预测概率"""
        super().predict_proba(X)
        
        # 获取所有子模型的概率预测
        probas = []
        for i, model in enumerate(self.models):
            proba = model.predict_proba(X)
            probas.append(proba[:, 1] * self.model_weights[i])  # 只取异常概率并加权
        
        # 计算加权平均概率
        ensemble_proba = np.sum(probas, axis=0)
        # 确保概率在[0,1]范围内
        ensemble_proba = np.clip(ensemble_proba, 0, 1)
        
        # 返回完整的概率矩阵
        return np.column_stack((1 - ensemble_proba, ensemble_proba))

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """获取特征重要性（加权平均）"""
        if not self.is_trained:
            return {}
        
        # 收集所有子模型的特征重要性
        all_importances = []
        for i, model in enumerate(self.models):
            importance = model.get_feature_importance()
            if importance is not None:
                all_importances.append((importance, self.model_weights[i]))
        
        if not all_importances:
            return None
        
        # 计算加权平均特征重要性
        feature_names = set()
        for importance, _ in all_importances:
            feature_names.update(importance.keys())
        
        ensemble_importance = {}
        for feature in feature_names:
            weighted_sum = 0.0
            weight_sum = 0.0
            for importance, weight in all_importances:
                if feature in importance:
                    weighted_sum += importance[feature] * weight
                    weight_sum += weight
            if weight_sum > 0:
                ensemble_importance[feature] = weighted_sum / weight_sum
        
        return ensemble_importance

    def save(self, file_path: str) -> None:
        """保存集成模型"""
        if not self.is_trained:
            self.logger.warning("保存未训练的集成模型")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 保存元数据
        metadata = self.get_metadata()
        metadata["model_weights"] = self.model_weights
        
        # 保存子模型
        sub_models = []
        for i, model in enumerate(self.models):
            # 根据子模型类型使用正确的文件扩展名
            if model.model_type in ["mlp", "lstm"]:
                sub_model_path = f"{file_path}_submodel_{i}.keras"
            else:
                sub_model_path = f"{file_path}_submodel_{i}.pkl"
                
            model.save(sub_model_path)
            sub_models.append({
                "type": model.model_type,
                "path": sub_model_path
            })
        
        metadata["sub_models"] = sub_models
        
        # 保存元数据
        with open(f"{file_path}.meta", "wb") as f:
            pickle.dump(metadata, f)
        
        self.logger.info(f"集成模型已保存到 {file_path}")

    @classmethod
    def load(cls, file_path: str) -> "EnsembleModel":
        """加载集成模型"""
        # 检查元数据文件是否存在
        meta_file = f"{file_path}.meta"
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"模型元数据文件不存在: {meta_file}")
        
        # 加载元数据
        with open(meta_file, "rb") as f:
            metadata = pickle.load(f)
        
        # 创建模型实例
        model = cls(model_type=metadata.get("model_type", "ensemble"), **metadata["params"])
        model.is_trained = metadata["is_trained"]
        model.feature_names = metadata["feature_names"]
        model.train_timestamp = metadata["train_timestamp"]
        model.metrics = metadata["metrics"]
        model.model_weights = metadata.get("model_weights", model.model_weights)
        
        # 加载子模型
        sub_models = metadata.get("sub_models", [])
        model.models = []
        for sub_model_info in sub_models:
            model_type = sub_model_info["type"]
            sub_model_path = sub_model_info["path"]
            
            # 根据模型类型加载子模型
            if model_type == "xgboost":
                sub_model = XGBoostModel.load(sub_model_path)
            elif model_type == "random_forest":
                sub_model = RandomForestModel.load(sub_model_path)
            elif model_type == "mlp":
                sub_model = MLPModel.load(sub_model_path)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            model.models.append(sub_model)
        
        model.logger.info(f"集成模型已从 {file_path} 加载")
        return model
