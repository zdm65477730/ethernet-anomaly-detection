import os
# 设置TensorFlow日志级别以屏蔽不必要的信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误和警告信息
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用oneDNN自定义操作

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, LSTM, Dropout, BatchNormalization, InputLayer
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from typing import Union, Optional, Dict
from .base_model import BaseModel

class LSTMModel(BaseModel):
    """LSTM模型（适合时序特征，如TCP流量的序列模式）"""
    def __init__(self,** kwargs):
        default_params = {
            "sequence_length": 30,  # 时序窗口长度
            "lstm_units": [64, 32],  # LSTM层单元数
            "dropout": 0.2,
            "batch_size": 32,
            "epochs": 20,
            "learning_rate": 0.001,
            "threshold": 0.5  # 可调分类阈值
        }
        params = {**default_params,** kwargs}
        super().__init__(**params)
        self.sequence_length = params["sequence_length"]
        self.model = None  # 延迟构建，需知道输入特征维度

    def _build_model(self) -> tf.keras.Model:
        """构建LSTM网络结构"""
        if "n_features" not in self.params:
            raise ValueError("特征维度未设置，请先调用fit或设置params['n_features']")
            
        model = Sequential(name="lstm_anomaly_detector")
        # 修复：使用shape参数替代已弃用的input_shape参数
        model.add(InputLayer(shape=(self.sequence_length, self.params["n_features"])))
        
        # 添加LSTM层 - 增强架构
        for i, units in enumerate(self.params["lstm_units"]):
            return_sequences = i != len(self.params["lstm_units"]) - 1
            model.add(LSTM(units, return_sequences=return_sequences, 
                          dropout=self.params["dropout"]/2, 
                          recurrent_dropout=self.params["dropout"]/2))
            model.add(BatchNormalization())
            # 添加额外的密集层以增强表达能力
            if not return_sequences:
                model.add(Dense(units//2, activation="relu"))
                model.add(Dropout(self.params["dropout"]))
        
        # 输出层（二分类）
        model.add(Dense(1, activation="sigmoid"))
        
        # 编译模型
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.params["learning_rate"])
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy", 
                     tf.keras.metrics.Precision(name="precision"), 
                     tf.keras.metrics.Recall(name="recall")]
        )
        return model

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],** kwargs
    ) -> None:
        """
        训练LSTM模型（需输入时序序列数据）

        参数:
            X: 形状为(n_samples, sequence_length, n_features)的时序特征
            y: 标签数据
        """
        super().fit(X, y, **kwargs)
        
        # 验证输入形状是否符合时序模型要求
        if len(X.shape) != 3:
            raise ValueError(f"LSTM输入必须是3维张量，实际形状: {X.shape}")
            
        self.params["n_features"] = X.shape[2]  # 记录特征维度
        self.model = self._build_model()
        
        self.logger.info(
            f"开始LSTM训练，样本数: {len(X)}, "
            f"序列长度: {self.sequence_length}, "
            f"特征数: {self.params['n_features']}"
        )
        
        # 保存训练数据用于特征重要性计算（只保存一部分以节省内存）
        # 只保存前1000个样本用于特征重要性计算，以节省内存
        max_samples = min(1000, len(X))
        if isinstance(X, pd.DataFrame):
            self._X_train = X.iloc[:max_samples].copy()
        else:
            self._X_train = X[:max_samples].copy()
        if isinstance(y, pd.Series):
            self._y_train = y.iloc[:max_samples].copy()
        else:
            self._y_train = y[:max_samples].copy()
        
        # 计算类别权重以处理不平衡数据
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))
        
        self.logger.info(f"类别权重: {class_weight_dict}")
        
        # 准备回调函数
        callbacks = [
            EarlyStopping(patience=3, monitor="val_loss", restore_best_weights=True),
            ModelCheckpoint(
                f"tmp_{self.model_type}_checkpoint.h5",
                monitor="val_precision",
                save_best_only=True,
                verbose=0
            )
        ]

        # 训练模型
        history = self.model.fit(
            X, y,
            batch_size=self.params["batch_size"],
            epochs=self.params["epochs"],
            validation_split=kwargs.get("validation_split", 0.2),
            callbacks=callbacks,
            shuffle=True,
            class_weight=class_weight_dict,  # 添加类别权重
            verbose=0
        )
        
        # 记录训练指标
        self.metrics = {
            "train_loss": history.history["loss"][-1],
            "val_loss": history.history["val_loss"][-1],
            "precision": history.history["precision"][-1],
            "recall": history.history["recall"][-1]
        }
        self.is_trained = True

    def train_on_batch(
        self,
        X_batch: Union[pd.DataFrame, np.ndarray],
        y_batch: Union[pd.Series, np.ndarray]
    ) -> dict:
        """批次训练（增量训练）"""
        if len(X_batch.shape) != 3:
            raise ValueError(f"LSTM批次输入必须是3维张量，实际形状: {X_batch.shape}")
            
        if not self.is_trained:
            self.params["n_features"] = X_batch.shape[2]
            self.model = self._build_model()  # 重新构建以适配特征维度
        
        loss, acc, precision, recall = self.model.train_on_batch(X_batch, y_batch)
        self.is_trained = True
        return {
            "loss": loss,
            "accuracy": acc,
            "precision": precision,
            "recall": recall
        }

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        super().predict(X)
        proba = self.model.predict(X, verbose=0).flatten()
        # 使用可调阈值而不是固定的0.5
        threshold = self.params.get("threshold", 0.5)
        return (proba >= threshold).astype(int)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        super().predict_proba(X)
        proba = self.model.predict(X, verbose=0).flatten()
        return np.column_stack((1 - proba, proba))  # [正常概率, 异常概率]

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """
        计算模型的准确率得分
        
        参数:
            X: 特征数据（3D张量）
            y: 真实标签
            
        返回:
            准确率得分
        """
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        获取特征重要性（使用Permutation Importance方法）
        注意：此方法需要在模型训练后调用，并且需要训练数据来计算重要性
        """
        if not self.is_trained:
            self.logger.warning("模型未训练，无法计算特征重要性")
            return {}
        
        # 检查是否已存储训练数据用于计算特征重要性
        if not hasattr(self, '_X_train') or not hasattr(self, '_y_train'):
            self.logger.info("LSTM模型未保存训练数据，无法计算Permutation Importance。"
                           "如需计算特征重要性，请在训练时保存训练数据。")
            return None
            
        try:
            # 导入permutation_importance函数
            from sklearn.inspection import permutation_importance
            
            # 使用permutation importance计算特征重要性
            # 这里使用模型自身的预测方法作为scoring函数
            perm_importance = permutation_importance(
                self, 
                self._X_train, 
                self._y_train, 
                n_repeats=10,  # 重复次数
                random_state=42,
                n_jobs=1  # 为了避免潜在的多线程问题
            )
            
            # 构建特征重要性字典
            if self.feature_names:
                importance_dict = dict(zip(self.feature_names, perm_importance.importances_mean))
            else:
                importance_dict = {f"feature_{i}": imp for i, imp in enumerate(perm_importance.importances_mean)}
            
            self.logger.info("成功计算LSTM模型的Permutation Importance特征重要性")
            return importance_dict
            
        except Exception as e:
            self.logger.warning(f"计算Permutation Importance时出错: {str(e)}")
            return None

    def save(self, file_path: str) -> None:
        """保存模型（含架构和权重）"""
        if not self.is_trained:
            self.logger.warning("保存未训练的LSTM模型")
        
        # 创建父目录
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 保存完整模型（架构+权重+配置）
        self.model.save(file_path)
        
        # 单独保存元数据
        metadata = self.get_metadata()
        with open(f"{file_path}.meta", "wb") as f:
            import pickle
            pickle.dump(metadata, f)
        
        self.logger.info(f"LSTM模型已保存到 {file_path}")

    @classmethod
    def load(cls, file_path: str) -> "LSTMModel":
        """从文件加载模型"""
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"模型文件不存在: {file_path}")
            
        meta_file = f"{file_path}.meta"
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"模型元数据文件不存在: {meta_file}")
        
        # 加载元数据
        with open(meta_file, "rb") as f:
            import pickle
            metadata = pickle.load(f)
        
        # 创建模型实例
        model = cls(** metadata["params"])
        model.is_trained = metadata["is_trained"]
        model.feature_names = metadata["feature_names"]
        model.train_timestamp = metadata["train_timestamp"]
        model.metrics = metadata["metrics"]
        
        # 加载模型权重
        model.model = tf.keras.models.load_model(file_path)
        model.logger.info(f"LSTM模型已从 {file_path} 加载")
        return model


class MLPModel(BaseModel):
    """多层感知器（适合结构化特征，如UDP流量的统计特征）"""
    def __init__(self, **kwargs):
        default_params = {
            "hidden_units": [128, 64, 32],  # 隐藏层单元数
            "dropout": 0.3,
            "batch_size": 64,
            "epochs": 30,
            "learning_rate": 0.001,
            "threshold": 0.5  # 可调分类阈值
        }
        params = {** default_params, **kwargs}
        super().__init__(** params)
        self.model = None  # 延迟构建，需知道输入特征维度

    def _build_model(self, n_features: int) -> tf.keras.Model:
        """构建MLP网络结构"""
        model = Sequential(name="mlp_anomaly_detector")
        model.add(InputLayer(shape=(n_features,)))
        
        # 添加隐藏层 - 增强架构
        for i, units in enumerate(self.params["hidden_units"]):
            model.add(Dense(units, activation="relu"))
            model.add(BatchNormalization())
            # 添加残差连接（仅当输入输出维度匹配时）
            if i > 0 and units == self.params["hidden_units"][i-1]:
                # 添加残差连接层
                model.add(tf.keras.layers.Add())
            model.add(Dropout(self.params["dropout"]))
        
        # 添加额外的密集层以增强表达能力
        model.add(Dense(16, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(self.params["dropout"]/2))  # 减少dropout率
        
        # 输出层
        model.add(Dense(1, activation="sigmoid"))
        
        # 编译 - 添加class_weight处理不平衡数据
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.params["learning_rate"])
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.Precision(name="Precision"), 
                     tf.keras.metrics.Recall(name="Recall")]
        )
        return model

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],** kwargs
    ) -> None:
        super().fit(X, y, **kwargs)
        n_features = X.shape[1]
        self.model = self._build_model(n_features)
        
        self.logger.info(f"开始MLP训练，样本数: {len(X)}, 特征数: {n_features}")
        
        # 保存训练数据用于特征重要性计算（只保存一部分以节省内存）
        # 只保存前1000个样本用于特征重要性计算，以节省内存
        max_samples = min(1000, len(X))
        if isinstance(X, pd.DataFrame):
            self._X_train = X.iloc[:max_samples].copy()
        else:
            self._X_train = X[:max_samples].copy()
        if isinstance(y, pd.Series):
            self._y_train = y.iloc[:max_samples].copy()
        else:
            self._y_train = y[:max_samples].copy()
        
        # 计算类别权重以处理不平衡数据
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))
        
        self.logger.info(f"类别权重: {class_weight_dict}")
        
        # 训练
        history = self.model.fit(
            X, y,
            batch_size=self.params["batch_size"],
            epochs=self.params["epochs"],
            validation_split=kwargs.get("validation_split", 0.2),
            callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
            class_weight=class_weight_dict,  # 添加类别权重
            verbose=0
        )
        
        # 修复：使用正确的指标名称（注意大小写）
        self.metrics = {
            "train_loss": history.history["loss"][-1],
            "val_loss": history.history["val_loss"][-1],
            "precision": history.history["Precision"][-1],  # 大写开头
            "recall": history.history["Recall"][-1]         # 大写开头
        }
        self.is_trained = True

    def train_on_batch(
        self,
        X_batch: Union[pd.DataFrame, np.ndarray],
        y_batch: Union[pd.Series, np.ndarray]
    ) -> dict:
        """批次训练"""
        if not self.is_trained:
            self.model = self._build_model(X_batch.shape[1])
        
        loss, acc, precision, recall = self.model.train_on_batch(X_batch, y_batch)
        self.is_trained = True
        return {
            "loss": loss,
            "accuracy": acc,
            "precision": precision,
            "recall": recall
        }

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        super().predict(X)
        proba = self.model.predict(X, verbose=0).flatten()
        # 使用可调阈值而不是固定的0.5
        threshold = self.params.get("threshold", 0.5)
        return (proba >= threshold).astype(int)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        super().predict_proba(X)
        proba = self.model.predict(X, verbose=0).flatten()
        return np.column_stack((1 - proba, proba))

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """
        计算模型的准确率得分
        
        参数:
            X: 特征数据
            y: 真实标签
            
        返回:
            准确率得分
        """
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        获取特征重要性（使用Permutation Importance方法）
        注意：此方法需要在模型训练后调用，并且需要训练数据来计算重要性
        """
        if not self.is_trained:
            self.logger.warning("模型未训练，无法计算特征重要性")
            return {}
        
        # 检查是否已存储训练数据用于计算特征重要性
        if not hasattr(self, '_X_train') or not hasattr(self, '_y_train'):
            self.logger.info("MLP模型未保存训练数据，无法计算Permutation Importance。"
                           "如需计算特征重要性，请在训练时保存训练数据。")
            return None
            
        try:
            # 导入permutation_importance函数
            from sklearn.inspection import permutation_importance
            
            # 使用permutation importance计算特征重要性
            # 这里使用模型自身的预测方法作为scoring函数
            perm_importance = permutation_importance(
                self, 
                self._X_train, 
                self._y_train, 
                n_repeats=10,  # 重复次数
                random_state=42,
                n_jobs=1  # 为了避免潜在的多线程问题
            )
            
            # 构建特征重要性字典
            if self.feature_names:
                importance_dict = dict(zip(self.feature_names, perm_importance.importances_mean))
            else:
                importance_dict = {f"feature_{i}": imp for i, imp in enumerate(perm_importance.importances_mean)}
            
            self.logger.info("成功计算MLP模型的Permutation Importance特征重要性")
            return importance_dict
            
        except Exception as e:
            self.logger.warning(f"计算Permutation Importance时出错: {str(e)}")
            return None

    def save(self, file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.model.save(file_path)
        # 保存元数据
        with open(f"{file_path}.meta", "wb") as f:
            import pickle
            pickle.dump(self.get_metadata(), f)
        self.logger.info(f"MLP模型已保存到 {file_path}")

    @classmethod
    def load(cls, file_path: str) -> "MLPModel":
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"模型文件不存在: {file_path}")
            
        meta_file = f"{file_path}.meta"
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"模型元数据文件不存在: {meta_file}")
        
        with open(meta_file, "rb") as f:
            import pickle
            metadata = pickle.load(f)
        
        model = cls(** metadata["params"])
        model.model = tf.keras.models.load_model(file_path)
        model.is_trained = metadata["is_trained"]
        model.feature_names = metadata["feature_names"]
        model.logger.info(f"MLP模型已从 {file_path} 加载")
        return model