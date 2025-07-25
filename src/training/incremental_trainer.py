import time
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from src.utils.logger import get_logger
from src.config.config_manager import ConfigManager
from src.models.model_factory import ModelFactory
from src.models.base_model import BaseModel
from .model_evaluator import ModelEvaluator

class IncrementalTrainer:
    """增量训练器，支持模型的增量更新和部分拟合"""
    
    def __init__(
        self,
        model_factory: ModelFactory,
        config: Optional[ConfigManager] = None,
        evaluator: Optional[ModelEvaluator] = None
    ):
        self.model_factory = model_factory
        self.config = config or ConfigManager()
        self.evaluator = evaluator or ModelEvaluator(config=self.config)
        self.logger = get_logger("training.incremental")
        
        # 增量训练配置
        self.increment_config = {
            "batch_size": self.config.get("training.incremental.batch_size", 1024),
            "min_samples": self.config.get("training.incremental.min_samples", 1000),
            "evaluation_interval": self.config.get("training.incremental.evaluation_interval", 5),
            "retrain_threshold": self.config.get("training.incremental.retrain_threshold", 0.1),
            "models_dir": self.config.get("model.models_dir", "models")
        }
        
        # 记录各模型的增量训练历史
        self.increment_history: Dict[str, List[Dict]] = {}
    
    def update_model_incrementally(
        self,
        model_type: str,
        new_X: Union[pd.DataFrame, np.ndarray],
        new_y: Union[pd.Series, np.ndarray],
        model: Optional[BaseModel] = None,
        protocol_labels: Optional[List[int]] = None,
        batch_size: Optional[int] = None,
        evaluation_interval: Optional[int] = None
    ) -> Tuple[BaseModel, Dict[str, float], bool]:
        """
        增量更新模型
        
        参数:
            model_type: 模型类型
            new_X: 新的特征数据
            new_y: 新的标签数据
            model: 已有的模型，为None则加载最新模型
            protocol_labels: 协议标签
            batch_size: 批次大小
            evaluation_interval: 评估间隔（批次）
            
        返回:
            更新后的模型、评估指标、是否触发了全量重训
        """
        start_time = time.time()
        batch_size = batch_size or self.increment_config["batch_size"]
        evaluation_interval = evaluation_interval or self.increment_config["evaluation_interval"]
        
        # 确保数据格式正确
        if isinstance(new_X, pd.DataFrame):
            X_np = new_X.values
            feature_names = new_X.columns.tolist()
        else:
            X_np = new_X
            feature_names = [f"feature_{i}" for i in range(X_np.shape[1])]
        
        if isinstance(new_y, pd.Series):
            y_np = new_y.values
        else:
            y_np = new_y
        
        # 检查样本数
        if len(X_np) < self.increment_config["min_samples"]:
            self.logger.info(
                f"新数据样本数 {len(X_np)} 小于最小阈值 {self.increment_config['min_samples']}, "
                "不进行增量更新"
            )
            return model, {}, False
        
        # 获取或加载模型
        if model is None:
            try:
                model = self.model_factory.load_latest_model(model_type)
                self.logger.info(f"已加载最新 {model_type} 模型用于增量更新")
            except Exception as e:
                self.logger.warning(f"加载最新模型失败: {str(e)}, 将训练新模型")
                from .model_trainer import ModelTrainer
                trainer = ModelTrainer(self.model_factory, self.config, self.evaluator)
                model, metrics, _ = trainer.train_new_model(model_type, X_np, y_np)
                return model, metrics, False
        
        # 检查模型是否支持增量训练
        if not hasattr(model, "partial_fit") and not hasattr(model, "train_on_batch"):
            self.logger.warning(
                f"{model_type} 不支持增量训练，将使用新数据全量重训"
            )
            from .model_trainer import ModelTrainer
            trainer = ModelTrainer(self.model_factory, self.config, self.evaluator)
            model, metrics, _ = trainer.train_new_model(model_type, X_np, y_np)
            return model, metrics, True
        
        # 分割评估集（用于增量训练后的评估）
        eval_size = min(int(0.1 * len(X_np)), 1000)  # 最多1000个样本用于评估
        X_train, X_eval, y_train, y_eval = self._split_data(
            X_np, y_np, test_size=eval_size
        )
        
        # 准备协议标签（如果提供）
        eval_protocol_labels = None
        if protocol_labels is not None:
            _, _, _, eval_protocol_labels = self._split_data(
                np.arange(len(protocol_labels)), protocol_labels, test_size=eval_size
            )
        
        # 记录初始性能
        initial_metrics = self._evaluate_incremental(
            model, model_type, X_eval, y_eval, eval_protocol_labels, feature_names
        )
        self.logger.info(
            f"增量更新前性能 - F1: {initial_metrics['f1']:.4f}, "
            f"Precision: {initial_metrics['precision']:.4f}, "
            f"Recall: {initial_metrics['recall']:.4f}"
        )
        
        # 计算批次数量
        num_batches = max(1, len(X_train) // batch_size)
        self.logger.info(
            f"开始增量训练，样本数: {len(X_train)}, 批次大小: {batch_size}, "
            f"批次数: {num_batches}"
        )
        
        # 记录训练历史
        if model_type not in self.increment_history:
            self.increment_history[model_type] = []
        
        # 按批次进行增量训练
        batch_metrics = []
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(X_train))
            
            X_batch = X_train[batch_start:batch_end]
            y_batch = y_train[batch_start:batch_end]
            
            # 执行增量训练
            batch_result = self._train_batch(model, X_batch, y_batch)
            batch_metrics.append(batch_result)
            
            # 定期评估
            if (i + 1) % evaluation_interval == 0 or (i + 1) == num_batches:
                eval_metrics = self._evaluate_incremental(
                    model, model_type, X_eval, y_eval, eval_protocol_labels, feature_names
                )
                self.logger.info(
                    f"批次 {i+1}/{num_batches} 完成 - "
                    f"训练损失: {batch_result.get('loss', 0):.4f}, "
                    f"评估F1: {eval_metrics['f1']:.4f}"
                )
        
        # 最终评估
        final_metrics = self._evaluate_incremental(
            model, model_type, X_eval, y_eval, eval_protocol_labels, feature_names
        )
        
        # 检查性能是否下降过多
        performance_drop = initial_metrics["f1"] - final_metrics["f1"]
        trigger_full_retrain = performance_drop > self.increment_config["retrain_threshold"]
        
        if trigger_full_retrain:
            self.logger.warning(
                f"增量训练后性能下降超过阈值 ({performance_drop:.4f} > "
                f"{self.increment_config['retrain_threshold']}), 将进行全量重训"
            )
            from .model_trainer import ModelTrainer
            trainer = ModelTrainer(self.model_factory, self.config, self.evaluator)
            model, final_metrics, _ = trainer.train_new_model(
                model_type, X_np, y_np, protocol_labels
            )
        
        # 保存更新后的模型
        model_path = self._save_incremental_model(model, model_type, final_metrics)
        
        # 记录增量训练历史
        self.increment_history[model_type].append({
            "timestamp": time.time(),
            "samples_used": len(X_train),
            "initial_metrics": initial_metrics,
            "final_metrics": final_metrics,
            "performance_drop": performance_drop,
            "triggered_retrain": trigger_full_retrain,
            "model_path": model_path,
            "batch_count": num_batches,
            "avg_batch_loss": np.mean([m.get("loss", 0) for m in batch_metrics])
        })
        
        self.logger.info(
            f"增量训练完成，耗时 {time.time() - start_time:.2f}秒，"
            f"最终F1: {final_metrics['f1']:.4f}"
        )
        
        return model, final_metrics, trigger_full_retrain
    
    def _split_data(self, X, y, test_size: int):
        """分割数据为训练集和评估集"""
        if test_size <= 0 or test_size >= len(X):
            return X, np.array([]), y, np.array([])
            
        # 随机选择评估样本
        indices = np.random.choice(len(X), size=test_size, replace=False)
        mask = np.zeros(len(X), dtype=bool)
        mask[indices] = True
        
        return X[~mask], X[mask], y[~mask], y[mask]
    
    def _train_batch(
        self,
        model: BaseModel,
        X_batch: np.ndarray,
        y_batch: np.ndarray
    ) -> Dict[str, float]:
        """训练单个批次"""
        try:
            # 优先使用partial_fit（传统模型）
            if hasattr(model, "partial_fit"):
                model.partial_fit(X_batch, y_batch)
                return {"success": True}
            
            # 其次使用train_on_batch（深度学习模型）
            elif hasattr(model, "train_on_batch"):
                result = model.train_on_batch(X_batch, y_batch)
                if isinstance(result, list) or isinstance(result, tuple):
                    # Keras模型通常返回 [loss, metric1, metric2, ...]
                    return {
                        "success": True,
                        "loss": result[0],
                        "accuracy": result[1] if len(result) > 1 else None
                    }
                return {"success": True, "loss": result}
                
        except Exception as e:
            self.logger.error(f"批次训练失败: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _evaluate_incremental(
        self,
        model: BaseModel,
        model_type: str,
        X_eval: np.ndarray,
        y_eval: np.ndarray,
        protocol_labels: Optional[List[int]],
        feature_names: List[str]
    ) -> Dict[str, float]:
        """评估增量训练效果"""
        if len(X_eval) == 0:
            return {"precision": 0, "recall": 0, "f1": 0}
            
        try:
            metrics, _ = self.evaluator.evaluate_model(
                model=model,
                model_type=model_type,
                X_test=X_eval,
                y_test=y_eval,
                protocol_labels=protocol_labels,
                feature_names=feature_names,
                output_path=None  # 增量评估不保存报告
            )
            return metrics
        except Exception as e:
            self.logger.error(f"增量评估失败: {str(e)}")
            return {"precision": 0, "recall": 0, "f1": 0}
    
    def _save_incremental_model(
        self,
        model: BaseModel,
        model_type: str,
        metrics: Dict[str, float]
    ) -> str:
        """保存增量训练后的模型"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        f1_score_str = f"{metrics['f1']:.4f}".replace(".", "_")
        model_filename = f"{model_type}_incremental_{timestamp}_f1_{f1_score_str}.pkl"
        model_path = os.path.join(self.increment_config["models_dir"], model_filename)
        
        model.save(model_path)
        
        # 更新最新模型链接
        self.model_factory.set_latest_model(model_type, model_path)
        
        self.logger.info(f"增量训练后的模型已保存至: {model_path}")
        return model_path
    
    def get_increment_history(self, model_type: Optional[str] = None) -> Dict[str, List[Dict]]:
        """获取增量训练历史"""
        if model_type:
            return {model_type: self.increment_history.get(model_type, [])}
        return self.increment_history.copy()
    
    def should_retrain_full(
        self,
        model_type: str,
        performance_drop_threshold: Optional[float] = None
    ) -> bool:
        """判断是否需要全量重训"""
        performance_drop_threshold = (
            performance_drop_threshold or self.increment_config["retrain_threshold"]
        )
        
        if model_type not in self.increment_history or len(self.increment_history[model_type]) == 0:
            return False
            
        # 检查最近几次增量训练的性能趋势
        recent_history = self.increment_history[model_type][-3:]  # 最近3次
        performance_drops = [h["performance_drop"] for h in recent_history]
        
        # 如果连续多次性能下降超过阈值，建议全量重训
        if len([d for d in performance_drops if d > performance_drop_threshold]) >= 2:
            return True
            
        return False
