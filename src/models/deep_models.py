import os
# 设置TensorFlow日志级别以屏蔽不必要的信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误和警告信息
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用oneDNN自定义操作

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, List, Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from typing_extensions import Self
from .base_model import BaseModel

class LSTMModel(BaseModel):
    """LSTM模型（适合时序特征，如TCP流量的序列模式）"""
    def __init__(self, model_type: str = "lstm", **kwargs):
        default_params = {
            "sequence_length": 30,  # 时序窗口长度
            "lstm_units": [64, 32],  # LSTM层单元数
            "dropout": 0.2,
            "batch_size": 32,
            "epochs": 20,
            "learning_rate": 0.001,
            "threshold": 0.5,  # 可调分类阈值
            "n_features": None  # 特征维度（延迟设置）
        }
        params = {**default_params, **kwargs}
        super().__init__(model_type, **params)
        
        # LSTM特定参数
        self.sequence_length = self.params.get("sequence_length", 30)
        self.lstm_units = self.params.get("lstm_units", [64, 32])
        self.dropout = self.params.get("dropout", 0.2)
        self.batch_size = self.params.get("batch_size", 32)
        self.epochs = self.params.get("epochs", 20)
        self.learning_rate = self.params.get("learning_rate", 0.001)
        self.threshold = self.params.get("threshold", 0.5)
        self.n_features = self.params.get("n_features")  # 特征维度
        
        # 初始化模型组件
        self.scaler = StandardScaler()
        self.model = None  # 延迟构建
        self._train_X = None
        self._train_y = None

    def _build_model(self, input_shape: tuple) -> tf.keras.Model:
        """
        构建LSTM模型
        
        参数:
            input_shape: 输入形状 (sequence_length, n_features)
        """
        from tensorflow.keras.layers import InputLayer
        
        model = tf.keras.Sequential([
            InputLayer(shape=input_shape),  # 使用InputLayer明确指定输入形状
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
        )
        
        return model

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray], **kwargs
    ) -> None:
        """
        训练LSTM模型（需输入时序序列数据）

        参数:
            X: 形状为(n_samples, n_features)的原始特征数据或(n_sequences, sequence_length, n_features)的序列数据
            y: 标签数据
        """
        super().fit(X, y, **kwargs)
        
        # 保存训练数据用于特征重要性计算
        if len(X.shape) == 2:
            self._train_X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
            self._train_y = y.copy() if isinstance(y, pd.Series) else pd.Series(y)
        elif len(X.shape) == 3:
            # 对于3D数据，我们只保存最后一个时间步的数据用于特征重要性计算
            self._train_X = pd.DataFrame(X[:, -1, :]) if not isinstance(X, pd.DataFrame) else X[:, -1, :]
            # 对于3D数据，y的长度可能与X的第一维不同，需要相应调整
            if isinstance(y, pd.Series):
                self._train_y = y.iloc[-X.shape[0]:].copy() if len(y) > X.shape[0] else y.copy()
            else:
                self._train_y = pd.Series(y[-X.shape[0]:]) if len(y) > X.shape[0] else pd.Series(y)
        
        # 确定特征维度
        if len(X.shape) == 2:
            self.n_features = X.shape[1]
        elif len(X.shape) == 3:
            self.n_features = X.shape[2]  # (n_samples, sequence_length, n_features)
        else:
            raise ValueError(f"输入特征必须是2维或3维，实际形状: {X.shape}")
        
        # 标准化特征
        if len(X.shape) == 2:
            X_scaled = self.scaler.fit_transform(X)
            
            # 创建时序序列
            X_seq, y_seq = self._create_sequences(X_scaled, y)
        elif len(X.shape) == 3:
            # 如果已经是3D格式，直接使用
            X_seq = X
            y_seq = y
            
            # 对最后一个时间步的数据进行标准化
            X_last_step = X_seq[:, -1, :]
            self.scaler.fit(X_last_step)
        
        # 构建模型
        if self.model is None:
            self.model = self._build_model((self.sequence_length, self.n_features))
        
        # 记录特征维度到参数中
        self.params["n_features"] = self.n_features
        
        self.logger.info(
            f"开始LSTM训练，样本数: {len(X_seq)}, "
            f"序列长度: {self.sequence_length}, "
            f"特征数: {self.n_features}"
        )
        
        # 保存训练数据用于特征重要性计算（只保存一部分以节省内存）
        # 只保存前1000个样本用于特征重要性计算，以节省内存
        max_samples = min(1000, len(X_seq))
        if isinstance(X_seq, np.ndarray):
            self._X_train = X_seq[:max_samples].copy()
        else:
            self._X_train = X_seq.iloc[:max_samples].copy() if len(X_seq) > max_samples else X_seq.copy()
            
        if isinstance(y_seq, np.ndarray):
            self._y_train = y_seq[:max_samples].copy() if len(y_seq) > max_samples else y_seq.copy()
        else:
            self._y_train = y_seq.iloc[:max_samples].copy() if len(y_seq) > max_samples else y_seq.copy()
        
        # 计算类别权重以处理不平衡数据
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_seq)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_seq)
        class_weight_dict = dict(zip(classes, class_weights))
        
        self.logger.info(f"类别权重: {class_weight_dict}")
        
        # 准备回调函数
        callbacks = [
            EarlyStopping(patience=3, monitor="val_loss", restore_best_weights=True),
            ModelCheckpoint(
                f"tmp_{self.model_type}_checkpoint.h5",
                monitor="val_precision",
                save_best_only=True,
                mode='max',  # 添加mode参数，precision越高越好
                verbose=0
            )
        ]

        # 训练模型
        history = self.model.fit(
            X_seq, y_seq,
            batch_size=self.batch_size,
            epochs=self.epochs,
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
            "precision": history.history["precision_1"][-1] if "precision_1" in history.history else history.history["precision"][-1],
            "recall": history.history["recall_1"][-1] if "recall_1" in history.history else history.history["recall"][-1]
        }
        self.is_trained = True

    def train_on_batch(
        self,
        X_batch: Union[pd.DataFrame, np.ndarray],
        y_batch: Union[pd.Series, np.ndarray]
    ) -> dict:
        """批次训练（增量训练）"""
        if len(X_batch.shape) != 2:
            raise ValueError(f"LSTM批次输入必须是2维，实际形状: {X_batch.shape}")
            
        # 标准化特征
        X_scaled = self.scaler.transform(X_batch)
        
        # 创建时序序列
        X_seq, y_seq = self._create_sequences(X_scaled, y_batch)
        
        if not self.is_trained:
            self.n_features = X_batch.shape[1]
            self.model = self._build_model((self.sequence_length, self.n_features))  # 重新构建以适配特征维度
        
        loss, acc, precision, recall = self.model.train_on_batch(X_seq, y_seq)
        self.is_trained = True
        return {
            "loss": loss,
            "accuracy": acc,
            "precision": precision,
            "recall": recall
        }

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        super().predict(X)
        
        # 处理3D输入数据
        if len(X.shape) == 3:
            # 对于3D数据，我们需要对每个时间步进行标准化
            X_reshaped = X.reshape(-1, X.shape[-1])  # 将3D数据重塑为2D
            X_scaled = self.scaler.transform(X_reshaped)
            X_scaled = X_scaled.reshape(X.shape)  # 重新整形为原始3D形状
            X_seq = X_scaled
        else:
            # 对于2D数据，直接标准化
            X_scaled = self.scaler.transform(X)
            
            # 创建时序序列（用零填充开始部分）
            if len(X_scaled) < self.sequence_length:
                # 如果数据不足一个序列长度，用零填充
                padding = np.zeros((self.sequence_length - len(X_scaled), X_scaled.shape[1]))
                X_padded = np.vstack([padding, X_scaled])
                X_seq = X_padded.reshape(1, self.sequence_length, X_scaled.shape[1])
            else:
                # 创建最后一个序列
                X_seq = X_scaled[-self.sequence_length:].reshape(1, self.sequence_length, X_scaled.shape[1])
        
        # 预测
        probabilities = self.model.predict(X_seq, verbose=0)
        predictions = (probabilities.flatten() > self.threshold).astype(int)
        
        # 扩展预测结果到原始长度
        full_predictions = np.full(len(X), predictions[-1])
        return full_predictions

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        super().predict_proba(X)
        
        # 处理3D输入数据
        if len(X.shape) == 3:
            # 对于3D数据，我们需要对每个时间步进行标准化
            X_reshaped = X.reshape(-1, X.shape[-1])  # 将3D数据重塑为2D
            X_scaled = self.scaler.transform(X_reshaped)
            X_scaled = X_scaled.reshape(X.shape)  # 重新整形为原始3D形状
            X_seq = X_scaled
        else:
            # 对于2D数据，直接标准化
            X_scaled = self.scaler.transform(X)
            
            # 创建时序序列（用零填充开始部分）
            if len(X_scaled) < self.sequence_length:
                # 如果数据不足一个序列长度，用零填充
                padding = np.zeros((self.sequence_length - len(X_scaled), X_scaled.shape[1]))
                X_padded = np.vstack([padding, X_scaled])
                X_seq = X_padded.reshape(1, self.sequence_length, X_scaled.shape[1])
            else:
                # 创建最后一个序列
                X_seq = X_scaled[-self.sequence_length:].reshape(1, self.sequence_length, X_scaled.shape[1])
        
        # 预测概率
        probabilities = self.model.predict(X_seq, verbose=0).flatten()
        
        # 构造二维概率数组 [P(negative), P(positive)]
        prob_array = np.column_stack([1 - probabilities, probabilities])
        
        # 扩展概率结果到原始长度
        full_probabilities = np.tile(prob_array[-1], (len(X), 1))
        return full_probabilities

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """
        计算模型的准确率得分
        
        参数:
            X: 特征数据（可以是2D或3D张量）
            y: 真实标签
            
        返回:
            准确率得分
        """
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        获取特征重要性（LSTM不直接支持特征重要性计算）
        
        返回:
            特征重要性字典，LSTM模型返回None
        """
        self.logger.info("LSTM模型不直接支持特征重要性计算，需要使用Permutation Importance等方法")
        return None

    def save(self, file_path: str) -> None:
        """保存模型到文件"""
        if not self.is_trained:
            self.logger.warning("模型未训练，无法保存")
            return
        
        # 创建目录
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 确保文件路径有正确的扩展名
        if not (file_path.endswith('.keras') or file_path.endswith('.h5')):
            file_path = file_path + '.keras'
        
        # 分别保存Keras模型和其它组件
        base_path = file_path.rsplit('.', 1)[0]  # 移除扩展名
        model_path = base_path + '_model.keras'
        self.model.save(model_path)
        
        # 保存其他组件
        import joblib
        components = {
            'model_type': self.model_type,
            'params': self.params,
            'scaler': self.scaler,
            'n_features': self.n_features,
            'sequence_length': self.sequence_length,
            'threshold': self.threshold,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'train_timestamp': self.train_timestamp,
            'metrics': self.metrics
        }
        components_path = base_path + '_components.pkl'
        joblib.dump(components, components_path)
        self.logger.info(f"模型已保存到 {file_path}")

    @classmethod
    def load(cls, file_path: str) -> "LSTMModel":
        """从文件加载模型"""
        import joblib
        
        # 确保文件路径有正确的扩展名
        if not (file_path.endswith('.keras') or file_path.endswith('.h5')):
            file_path = file_path + '.keras'
        
        # 构造组件和模型文件路径
        base_path = file_path.rsplit('.', 1)[0]  # 移除扩展名
        components_path = base_path + '_components.pkl'
        model_path = base_path + '_model.keras'
        
        # 加载组件
        components = joblib.load(components_path)
        
        # 创建模型实例
        model = cls(model_type=components['model_type'], **components['params'])
        model.scaler = components['scaler']
        model.n_features = components['n_features']
        model.sequence_length = components['sequence_length']
        model.threshold = components['threshold']
        model.is_trained = components['is_trained']
        model.feature_names = components['feature_names']
        model.train_timestamp = components['train_timestamp']
        model.metrics = components['metrics']
        
        # 加载Keras模型
        model.model = tf.keras.models.load_model(model_path)
        
        return model


class MLPModel(BaseModel):
    """多层感知器（适合结构化特征，如UDP流量的统计特征）"""
    def __init__(self, **kwargs):
        # 从kwargs中提取model_type，如果不存在则默认为"mlp"
        model_type = kwargs.pop("model_type", "mlp")
        
        default_params = {
            "hidden_units": [128, 64, 32],  # 隐藏层单元数
            "dropout": 0.3,
            "batch_size": 64,
            "epochs": 30,
            "learning_rate": 0.001,
            "threshold": 0.5  # 可调分类阈值
        }
        params = {**default_params, **kwargs}
        super().__init__(model_type=model_type, ** params)
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
        
        # 处理3D输入数据（可能来自LSTM数据转换）
        if len(X.shape) == 3:
            # 如果是3D数据，我们只使用最后一个时间步的数据
            X = X[:, -1, :]  # 取最后一个时间步
            
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