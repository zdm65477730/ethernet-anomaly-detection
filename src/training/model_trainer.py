import time
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import StratifiedKFold, train_test_split
from src.utils.logger import get_logger
from src.config.config_manager import ConfigManager
from src.models.model_factory import ModelFactory
from src.models.base_model import BaseModel
from .model_evaluator import ModelEvaluator

class ModelTrainer:
    """基础模型训练器，实现完整的模型训练流程"""
    
    def __init__(
        self,
        model_factory: ModelFactory,
        config: Optional[ConfigManager] = None,
        evaluator: Optional[ModelEvaluator] = None
    ):
        self.model_factory = model_factory
        self.config = config or ConfigManager()
        self.evaluator = evaluator or ModelEvaluator(config=self.config)
        self.logger = get_logger("training.trainer")
        
        # 训练配置
        self.default_train_params = {
            "test_size": self.config.get("training.test_size", 0.2),
            "cv_folds": self.config.get("training.cross_validation_folds", 5),
            "random_state": self.config.get("training.random_state", 42),
            "models_dir": self.config.get("model.models_dir", "models")
        }
        
        # 创建模型保存目录
        os.makedirs(self.default_train_params["models_dir"], exist_ok=True)
    
    def train_new_model(
        self,
        model_type: str,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        protocol_labels: Optional[List[int]] = None,
        test_size: float = None,
        cv_folds: int = None,
        train_params: Optional[Dict] = None,
        output_dir: Optional[str] = None
    ) -> Tuple[BaseModel, Dict[str, float], str]:
        """
        训练新模型，包含数据分割、交叉验证和评估
        
        参数:
            model_type: 模型类型
            X: 特征数据
            y: 标签数据
            protocol_labels: 每个样本的协议标签
            test_size: 测试集比例
            cv_folds: 交叉验证折数
            train_params: 模型训练参数
            output_dir: 模型输出目录
            
        返回:
            训练好的模型、评估指标、模型保存路径
        """
        start_time = time.time()
        
        # 解析参数
        test_size = test_size or self.default_train_params["test_size"]
        cv_folds = cv_folds or self.default_train_params["cv_folds"]
        train_params = train_params or {}
        output_dir = output_dir or self.default_train_params["models_dir"]
        
        # 确保数据格式正确
        if isinstance(X, pd.DataFrame):
            X_np = X.values
            feature_names = X.columns.tolist()
        else:
            X_np = X
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            y_np = y.values
        else:
            y_np = y
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_np,
            test_size=test_size,
            random_state=self.default_train_params["random_state"],
            stratify=y_np  # 保持分层抽样
        )
        
        self.logger.info(
            f"数据分割完成 - 训练集: {len(X_train)} 样本, "
            f"测试集: {len(X_test)} 样本, 异常比例: {np.mean(y_np):.2%}"
        )
        
        # 执行交叉验证
        cv_metrics = self._cross_validate(
            model_type=model_type,
            X=X_train,
            y=y_train,
            folds=cv_folds,
            train_params=train_params
        )
        
        # 训练最终模型（使用全部训练数据）
        model = self.model_factory.create_model(model_type, **train_params)
        self.logger.info(f"开始训练最终 {model_type} 模型...")
        
        model.fit(X_train, y_train)
        
        # 评估最终模型
        test_protocol_labels = None
        if protocol_labels is not None:
            # 分割协议标签以匹配测试集
            _, _, _, test_protocol_labels = train_test_split(
                np.arange(len(protocol_labels)), protocol_labels,
                test_size=test_size,
                random_state=self.default_train_params["random_state"],
                stratify=y_np
            )
        
        test_metrics, report_path = self.evaluator.evaluate_model(
            model=model,
            model_type=model_type,
            X_test=X_test,
            y_test=y_test,
            protocol_labels=test_protocol_labels,
            feature_names=feature_names
        )
        
        # 保存模型
        model_path = self._save_trained_model(
            model=model,
            model_type=model_type,
            metrics=test_metrics,
            output_dir=output_dir
        )
        
        # 记录训练时间
        elapsed_time = time.time() - start_time
        self.logger.info(
            f"{model_type} 模型训练完成，耗时 {elapsed_time:.2f}秒，"
            f"测试集F1: {test_metrics['f1']:.4f}"
        )
        
        # 记录交叉验证和测试集指标
        model.metrics = {
            "cv": cv_metrics,
            "test": test_metrics,
            "training_time": elapsed_time
        }
        
        return model, test_metrics, model_path
    
    def _cross_validate(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        folds: int = 5,
        train_params: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        执行交叉验证
        
        参数:
            model_type: 模型类型
            X: 特征数据
            y: 标签数据
            folds: 交叉验证折数
            train_params: 训练参数
            
        返回:
            交叉验证的平均指标
        """
        if folds < 2:
            self.logger.warning("交叉验证折数小于2，跳过交叉验证")
            return {}
            
        train_params = train_params or {}
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.default_train_params["random_state"])
        
        # 存储每折的指标
        fold_metrics = {
            "precision": [],
            "recall": [],
            "f1": [],
            "auc": []
        }
        
        self.logger.info(f"开始 {folds} 折交叉验证训练 {model_type} 模型")
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            fold_start = time.time()
            
            # 分割数据
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            # 创建并训练模型
            model = self.model_factory.create_model(model_type,** train_params)
            model.fit(X_fold_train, y_fold_train)
            
            # 评估
            y_pred = model.predict(X_fold_val)
            try:
                y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            except NotImplementedError:
                y_pred_proba = None
            
            # 计算指标
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
            fold_precision = precision_score(y_fold_val, y_pred)
            fold_recall = recall_score(y_fold_val, y_pred)
            fold_f1 = f1_score(y_fold_val, y_pred)
            
            # 存储指标
            fold_metrics["precision"].append(fold_precision)
            fold_metrics["recall"].append(fold_recall)
            fold_metrics["f1"].append(fold_f1)
            
            # AUC（如果支持）
            if y_pred_proba is not None:
                try:
                    fold_auc = roc_auc_score(y_fold_val, y_pred_proba)
                    fold_metrics["auc"].append(fold_auc)
                except ValueError:
                    pass
            
            self.logger.debug(
                f"折 {fold+1}/{folds} 完成，耗时 {time.time() - fold_start:.2f}秒，"
                f"F1: {fold_f1:.4f}"
            )
        
        # 计算平均指标
        cv_metrics = {
            "precision": np.mean(fold_metrics["precision"]),
            "recall": np.mean(fold_metrics["recall"]),
            "f1": np.mean(fold_metrics["f1"]),
            "std_f1": np.std(fold_metrics["f1"])  # F1标准差
        }
        
        if fold_metrics["auc"]:
            cv_metrics["auc"] = np.mean(fold_metrics["auc"])
        
        self.logger.info(
            f"交叉验证完成 - 平均F1: {cv_metrics['f1']:.4f} "
            f"(±{cv_metrics['std_f1']:.4f})"
        )
        
        return cv_metrics
    
    def _save_trained_model(
        self,
        model: BaseModel,
        model_type: str,
        metrics: Dict[str, float],
        output_dir: str
    ) -> str:
        """
        保存训练好的模型
        
        参数:
            model: 训练好的模型
            model_type: 模型类型
            metrics: 模型评估指标
            output_dir: 输出目录
            
        返回:
            模型保存路径
        """
        # 生成模型文件名（包含时间戳和F1分数）
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        f1_score_str = f"{metrics['f1']:.4f}".replace(".", "_")
        model_filename = f"{model_type}_{timestamp}_f1_{f1_score_str}.pkl"
        model_path = os.path.join(output_dir, model_filename)
        
        # 保存模型
        model.save(model_path)
        
        # 更新模型工厂的最新模型链接
        self.model_factory.set_latest_model(model_type, model_path)
        
        self.logger.info(f"模型已保存至: {model_path}")
        return model_path
    
    def train_protocol_specific_models(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        protocol_labels: List[int],
        default_model_type: str = "xgboost",
        protocol_model_map: Optional[Dict[int, str]] = None
    ) -> Dict[int, Tuple[BaseModel, Dict[str, float]]]:
        """
        为不同协议训练专用模型
        
        参数:
            X: 特征数据
            y: 标签数据
            protocol_labels: 每个样本的协议标签
            default_model_type: 默认模型类型
            protocol_model_map: 协议到模型类型的映射
            
        返回:
            协议到(模型, 指标)的字典
        """
        protocol_model_map = protocol_model_map or {}
        unique_protocols = np.unique(protocol_labels)
        results = {}
        
        self.logger.info(f"开始为 {len(unique_protocols)} 个协议训练专用模型")
        
        for proto in unique_protocols:
            # 筛选该协议的样本
            mask = np.array(protocol_labels) == proto
            X_proto = X[mask]
            y_proto = y[mask]
            
            # 跳过样本数过少的协议
            if len(X_proto) < self.config.get("training.min_samples_per_protocol", 100):
                self.logger.debug(
                    f"协议 {proto} 样本数不足 ({len(X_proto)}), 跳过训练专用模型"
                )
                continue
            
            # 确定模型类型
            model_type = protocol_model_map.get(proto, default_model_type)
            proto_name = self._get_protocol_name(proto)
            
            self.logger.info(
                f"训练 {proto_name} 协议专用模型 ({model_type}), 样本数: {len(X_proto)}"
            )
            
            # 训练模型
            try:
                model, metrics, _ = self.train_new_model(
                    model_type=model_type,
                    X=X_proto,
                    y=y_proto,
                    output_dir=os.path.join(self.default_train_params["models_dir"], proto_name)
                )
                
                results[proto] = (model, metrics)
                
                self.logger.info(
                    f"{proto_name} 协议模型训练完成, F1: {metrics['f1']:.4f}"
                )
                
            except Exception as e:
                self.logger.error(
                    f"训练 {proto_name} 协议专用模型失败: {str(e)}",
                    exc_info=True
                )
        
        return results
    
    def _get_protocol_name(self, protocol_number: int) -> str:
        """获取协议名称（辅助函数）"""
        try:
            from src.features.protocol_specs import get_protocol_name
            return get_protocol_name(protocol_number)
        except:
            return f"proto_{protocol_number}"
