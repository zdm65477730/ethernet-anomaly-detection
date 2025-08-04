import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from typing import Union, Optional, Dict
import time
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    """XGBoost模型（适合统计特征与结构化数据）"""
    def __init__(self,** kwargs):
        # 从kwargs中提取model_type，如果不存在则默认为"xgboost"
        model_type = kwargs.pop("model_type", "xgboost")
        
        default_params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "scale_pos_weight": 10,  # 平衡异常样本少的问题
            "eval_metric": "logloss"
        }
        # 合并默认参数与用户参数
        params = {**default_params,** kwargs}
        # 移除XGBoost不支持的参数
        xgb_params = params.copy()
        xgb_params.pop("model_type", None)
        xgb_params.pop("use_label_encoder", None)
        
        super().__init__(model_type=model_type, **params)
        self.model = XGBClassifier(**xgb_params)

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],** kwargs
    ) -> None:
        super().fit(X, y, **kwargs)
        self.logger.info(f"开始XGBoost全量训练，样本数: {len(X)}")
        self.model.fit(X, y,** kwargs)
        self.is_trained = True
        # 计算训练指标
        if "eval_set" in kwargs:
            y_pred = self.predict(kwargs["eval_set"][0][0])
            y_true = kwargs["eval_set"][0][1]
            self.metrics = self._calculate_metrics(y_true, y_pred)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        super().predict(X)
        return self.model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        super().predict_proba(X)
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """获取特征重要性"""
        if not self.is_trained:
            return {}
        importances = self.model.feature_importances_
        if self.feature_names:
            return dict(zip(self.feature_names, importances))
        return {f"feature_{i}": imp for i, imp in enumerate(importances)}

    def save(self, file_path: str) -> None:
        """保存模型到文件"""
        if not self.is_trained:
            self.logger.warning("保存未训练的模型")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
        self.logger.info(f"XGBoost模型已保存到 {file_path}")

    @classmethod
    def load(cls, file_path: str) -> "XGBoostModel":
        """从文件加载模型"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"模型文件不存在: {file_path}")
            
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        model.logger.info(f"XGBoost模型已从 {file_path} 加载")
        return model

    def _calculate_metrics(self, y_true, y_pred):
        """计算评估指标"""
        return {
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_pred)
        }


class RandomForestModel(BaseModel):
    """随机森林模型（适合高维特征与非线性关系）"""
    def __init__(self, **kwargs):
        # 从kwargs中提取model_type，如果不存在则默认为"random_forest"
        model_type = kwargs.pop("model_type", "random_forest")
        
        default_params = {
            "n_estimators": 100,
            "max_depth": 8,
            "class_weight": "balanced",
            "n_jobs": -1
        }
        # 合并默认参数与用户参数
        params = {**default_params, **kwargs}
        super().__init__(model_type=model_type, **params)
        self.model = RandomForestClassifier(**params)

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],** kwargs
    ) -> None:
        super().fit(X, y, **kwargs)
        self.logger.info(f"开始随机森林全量训练，样本数: {len(X)}")
        self.model.fit(X, y,** kwargs)
        self.is_trained = True

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        super().predict(X)
        return self.model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        super().predict_proba(X)
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if not self.is_trained:
            return {}
        importances = self.model.feature_importances_
        if self.feature_names:
            return dict(zip(self.feature_names, importances))
        return {f"feature_{i}": imp for i, imp in enumerate(importances)}

    def save(self, file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
        self.logger.info(f"随机森林模型已保存到 {file_path}")

    @classmethod
    def load(cls, file_path: str) -> "RandomForestModel":
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"模型文件不存在: {file_path}")
            
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        model.logger.info(f"随机森林模型已从 {file_path} 加载")
        return model


class LogisticRegressionModel(BaseModel):
    """逻辑回归模型（支持增量训练）"""
    def __init__(self, **kwargs):
        default_params = {
            "loss": "log_loss",  # 逻辑回归损失
            "penalty": "l2",
            "alpha": 0.001,
            "class_weight": "balanced",
            "max_iter": 1000,
            "warm_start": True  # 支持增量训练
        }
        params = {** default_params, **kwargs}
        super().__init__(** params)
        self.model = SGDClassifier(**params)

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],** kwargs
    ) -> None:
        super().fit(X, y, **kwargs)
        self.logger.info(f"开始逻辑回归全量训练，样本数: {len(X)}")
        self.model.fit(X, y,** kwargs)
        self.is_trained = True

    def partial_fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],**kwargs
    ) -> None:
        """增量训练实现"""
        if not self.is_trained:
            # 首次增量训练需要初始化
            self.feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
            self.model.partial_fit(X, y, classes=[0, 1],** kwargs)
            self.is_trained = True
            self.logger.info("逻辑回归首次增量训练完成")
        else:
            self.model.partial_fit(X, y, **kwargs)
            self.logger.info(f"逻辑回归增量训练完成，新增样本数: {len(X)}")
        self.train_timestamp = time.time()

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        super().predict(X)
        return self.model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        super().predict_proba(X)
        # SGDClassifier无predict_proba，手动计算
        decision_func = self.model.decision_function(X)
        prob_pos = 1 / (1 + np.exp(-decision_func))  # sigmoid转换
        return np.column_stack((1 - prob_pos, prob_pos))

    def save(self, file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
        self.logger.info(f"逻辑回归模型已保存到 {file_path}")

    @classmethod
    def load(cls, file_path: str) -> "LogisticRegressionModel":
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"模型文件不存在: {file_path}")
            
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        model.logger.info(f"逻辑回归模型已从 {file_path} 加载")
        return model