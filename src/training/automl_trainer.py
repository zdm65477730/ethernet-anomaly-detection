import time
import os
import threading
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from src.utils.logger import get_logger
from src.config.config_manager import ConfigManager
from src.data.data_storage import DataStorage
from src.models.model_factory import ModelFactory
from src.models.model_selector import ModelSelector
from src.features.protocol_specs import get_protocol_spec, get_all_protocols
from .model_trainer import ModelTrainer
from .incremental_trainer import IncrementalTrainer
from .model_evaluator import ModelEvaluator
from .feedback_optimizer import FeedbackOptimizer

class AutoMLTrainer:
    """自动化机器学习训练器，实现完整的训练-评估-优化-再训练闭环"""
    
    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        data_storage: Optional[DataStorage] = None,
        model_factory: Optional[ModelFactory] = None,
        model_selector: Optional[ModelSelector] = None
    ):
        self.config = config or ConfigManager()
        self.data_storage = data_storage or DataStorage(config=self.config)
        self.model_factory = model_factory or ModelFactory(config=self.config)
        self.model_selector = model_selector or ModelSelector(config=self.config)
        
        # 初始化训练器组件
        self.evaluator = ModelEvaluator(config=self.config)
        self.feedback_optimizer = FeedbackOptimizer(config=self.config)
        self.base_trainer = ModelTrainer(
            model_factory=self.model_factory,
            config=self.config,
            evaluator=self.evaluator
        )
        self.incremental_trainer = IncrementalTrainer(
            model_factory=self.model_factory,
            config=self.config,
            evaluator=self.evaluator
        )
        
        self.logger = get_logger("training.automl")
        
        # AutoML配置
        self.automl_config = {
            "max_iterations": self.config.get("training.automl.max_iterations", 10),
            "target_f1_score": self.config.get("training.automl.target_f1_score", 0.9),
            "improvement_threshold": self.config.get("training.automl.improvement_threshold", 0.01),
            "max_training_time": self.config.get("training.automl.max_training_time", 86400),  # 24小时
            "enable_feature_optimization": self.config.get("training.automl.enable_feature_optimization", True),
            "enable_model_optimization": self.config.get("training.automl.enable_model_optimization", True),
            "enable_hyperparameter_tuning": self.config.get("training.automl.enable_hyperparameter_tuning", True),
        }
        
        # 状态变量
        self._is_running = False
        self._thread = None
        self.iteration_history: List[Dict] = []
        self.best_model_info: Optional[Dict] = None
        
        self.logger.info("AutoML训练器初始化完成")
    
    def start_automl_process(
        self,
        data_path: Optional[str] = None,
        protocol: Optional[int] = None,
        target_model_type: Optional[str] = None,
        background: bool = False
    ) -> bool:
        """
        启动AutoML训练过程
        
        参数:
            data_path: 训练数据路径
            protocol: 协议类型
            target_model_type: 目标模型类型
            background: 是否后台运行
            
        返回:
            是否成功启动
        """
        if self._is_running:
            self.logger.warning("AutoML训练已在运行中")
            return False
        
        if background:
            self._thread = threading.Thread(
                target=self._automl_training_loop,
                args=(data_path, protocol, target_model_type),
                daemon=True
            )
            self._thread.start()
            self.logger.info("AutoML训练已在后台启动")
        else:
            self._automl_training_loop(data_path, protocol, target_model_type)
        
        return True
    
    def stop_automl_process(self) -> None:
        """停止AutoML训练过程"""
        if not self._is_running:
            self.logger.warning("AutoML训练未在运行")
            return
        
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                self.logger.warning("AutoML训练线程未能正常终止")
        
        self.logger.info("AutoML训练已停止")
    
    def _automl_training_loop(
        self,
        data_path: Optional[str],
        protocol: Optional[int],
        target_model_type: Optional[str]
    ) -> None:
        """AutoML训练主循环"""
        self._is_running = True
        start_time = time.time()
        
        self.logger.info("开始AutoML训练过程")
        
        try:
            # 加载数据
            X, y, protocol_labels, feature_names = self._load_training_data(data_path)
            
            # 检查数据是否成功加载
            if len(X) == 0 or len(y) == 0:
                self.logger.error("未能成功加载训练数据")
                return
            
            # 如果没有指定协议，自动检测数据中的协议类型
            if protocol is None and protocol_labels is not None:
                self.logger.info("自动检测训练数据中的协议类型")
                unique_protocols = np.unique(protocol_labels)
                self.logger.info(f"检测到以下协议类型: {unique_protocols}")
                
                # 为每种协议分别训练模型
                for proto in unique_protocols:
                    if not self._is_running:
                        break
                    
                    proto_spec = get_protocol_spec(proto)
                    proto_name = proto_spec["name"]
                    self.logger.info(f"开始为 {proto_name} 协议训练模型")
                    
                    # 筛选出该协议的数据
                    mask = np.array(protocol_labels) == proto
                    X_proto = X[mask]
                    y_proto = y[mask]
                    
                    self.logger.info(f"{proto_name} 协议数据量: {len(X_proto)} 样本")
                    
                    # 为该协议选择最佳模型类型
                    model_type = self.model_selector.select_best_model(
                        protocol=proto,
                        candidates=["xgboost", "random_forest", "lstm", "mlp"]
                    )
                    
                    self.logger.info(f"为 {proto_name} 协议选择模型类型: {model_type}")
                    
                    # 训练该协议的模型
                    self._train_protocol_model(model_type, X_proto, y_proto, proto, feature_names)
                
                # 训练通用模型（使用所有数据）
                self.logger.info("开始训练通用模型")
                default_model_type = self.config.get("training.default_model", "xgboost")
                final_model_type = target_model_type or default_model_type
                
                self.logger.info(f"通用模型类型: {final_model_type}")
                self._train_protocol_model(final_model_type, X, y, None, feature_names)
            else:
                # 确定模型类型
                default_model_type = self.config.get("training.default_model", "xgboost")
                final_model_type = target_model_type or default_model_type
                
                if protocol is not None:
                    final_model_type = self.model_selector.select_best_model(
                        protocol=protocol,
                        candidates=["xgboost", "random_forest", "lstm", "mlp"]
                    )
                
                self.logger.info(f"选择模型类型: {final_model_type}")
                
                # 训练模型
                self._train_single_model(final_model_type, X, y, protocol, feature_names)
            
            self.logger.info("AutoML训练过程完成")
            
        except Exception as e:
            self.logger.error(f"AutoML训练过程中出错: {str(e)}", exc_info=True)
        finally:
            self._is_running = False
    
    def _train_single_model(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        protocol: Optional[int],
        feature_names: List[str]
    ) -> None:
        """训练单个模型"""
        start_time = time.time()
        
        # 初始化最佳模型信息
        self.best_model_info = {
            "iteration": 0,
            "model_type": model_type,
            "f1_score": 0.0,
            "metrics": {}
        }
        
        # 迭代训练和优化
        for iteration in range(1, self.automl_config["max_iterations"] + 1):
            if not self._is_running:
                break
            
            current_time = time.time()
            if current_time - start_time > self.automl_config["max_training_time"]:
                self.logger.warning("达到最大训练时间限制，停止训练")
                break
            
            self.logger.info(f"开始第 {iteration} 轮训练")
            
            # 训练模型
            model, metrics = self._train_model(
                model_type, X, y, [protocol]*len(X) if protocol else None, iteration
            )
            
            # 记录迭代历史
            iteration_info = {
                "iteration": iteration,
                "model_type": model_type,
                "metrics": metrics,
                "timestamp": current_time
            }
            self.iteration_history.append(iteration_info)
            
            # 检查是否达到目标
            current_f1 = metrics.get("f1", 0)
            if current_f1 >= self.automl_config["target_f1_score"]:
                self.logger.info(f"达到目标F1分数 {self.automl_config['target_f1_score']}，停止训练")
                self._save_best_model(model, model_type, metrics, iteration)
                break
            
            # 检查是否是最佳模型
            if current_f1 > self.best_model_info["f1_score"]:
                improvement = current_f1 - self.best_model_info["f1_score"]
                self.best_model_info.update({
                    "iteration": iteration,
                    "f1_score": current_f1,
                    "metrics": metrics
                })
                self.logger.info(f"发现更好的模型，F1分数提升 {improvement:.4f}")
                self._save_best_model(model, model_type, metrics, iteration)
            
            # 检查改进是否足够
            if len(self.iteration_history) >= 2:
                prev_f1 = self.iteration_history[-2]["metrics"].get("f1", 0)
                improvement = current_f1 - prev_f1
                if improvement < self.automl_config["improvement_threshold"]:
                    self.logger.info(
                        f"改进幅度 {improvement:.4f} 低于阈值 {self.automl_config['improvement_threshold']}，"
                        f"停止训练"
                    )
                    break
            
            # 执行优化
            if self.automl_config["enable_feature_optimization"] or \
               self.automl_config["enable_model_optimization"]:
                self._perform_optimization(model, model_type, metrics, protocol)
            
            # 如果启用了超参数调优，调整模型类型或参数
            if self.automl_config["enable_hyperparameter_tuning"]:
                model_type = self._tune_hyperparameters(
                    model_type, metrics, iteration
                )
    
    def _train_protocol_model(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        protocol: Optional[int],
        feature_names: List[str]
    ) -> None:
        """为特定协议训练模型"""
        try:
            # 简化版本，直接训练一次
            model, metrics, _ = self.base_trainer.train_new_model(
                model_type=model_type,
                X=X,
                y=y
            )
            
            # 保存模型
            model_path = self.model_factory.save_model(model, model_type)
            self.logger.info(f"保存 {model_type} 模型到: {model_path}")
            
            # 更新模型选择器的性能记录
            if protocol is not None:
                self.model_selector.update_performance(
                    protocol=protocol,
                    model_type=model_type,
                    metrics=metrics
                )
            
            self.logger.info(f"{model_type} 模型训练完成，F1分数: {metrics.get('f1', 0):.4f}")
            
        except Exception as e:
            self.logger.error(f"训练 {model_type} 模型时出错: {str(e)}", exc_info=True)
    
    def _load_training_data(self, data_path: Optional[str]):
        """加载训练数据"""
        try:
            # 如果没有指定路径，使用默认数据目录
            if not data_path:
                data_path = self.config.get("data.processed_dir", "data/processed")
            
            # 确保data_path不是None
            if not data_path:
                data_path = "data/processed"
            
            # 查找数据文件
            data_files = []
            if os.path.isdir(data_path):
                for file in os.listdir(data_path):
                    if file.endswith(".csv"):
                        data_files.append(os.path.join(data_path, file))
            elif os.path.isfile(data_path) and data_path.endswith(".csv"):
                data_files.append(data_path)
            
            if not data_files:
                self.logger.error(f"在路径 {data_path} 中未找到CSV数据文件")
                return np.array([]), np.array([]), None, []
            
            # 加载数据
            dfs = []
            for file in data_files:
                df = pd.read_csv(file)
                dfs.append(df)
            
            # 合并所有数据
            full_df = pd.concat(dfs, ignore_index=True)
            self.logger.info(f"总共加载 {len(full_df)} 条数据记录")
            
            # 分离特征和标签
            if 'label' not in full_df.columns:
                self.logger.error("数据中未找到'label'列作为标签")
                return np.array([]), np.array([]), None, []
            
            # 假设第一列是标签，其余是特征
            y = full_df['label'].values
            feature_columns = [col for col in full_df.columns if col != 'label']
            X = full_df[feature_columns].values
            feature_names = feature_columns
            
            # 尝试获取协议标签
            protocol_labels = None
            if 'protocol' in full_df.columns:
                protocol_labels = full_df['protocol'].tolist()
                self.logger.info(f"检测到协议标签，共 {len(set(protocol_labels))} 种协议")
            elif 'protocol_num' in full_df.columns:
                protocol_labels = full_df['protocol_num'].tolist()
                self.logger.info(f"检测到协议标签，共 {len(set(protocol_labels))} 种协议")
            
            return X, y, protocol_labels, feature_names
            
        except Exception as e:
            self.logger.error(f"加载训练数据时出错: {str(e)}", exc_info=True)
            return np.array([]), np.array([]), None, []
    
    def _train_model(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        protocol_labels: Optional[List[int]],
        iteration: int
    ) -> Tuple[Any, Dict[str, float]]:
        """训练模型"""
        try:
            # 如果不是第一轮，尝试增量训练
            if iteration > 1:
                try:
                    model = self.model_factory.load_latest_model(model_type)
                    model, metrics, _ = self.incremental_trainer.update_model_incrementally(
                        model_type=model_type,
                        new_X=X,
                        new_y=y,
                        model=model,
                        protocol_labels=protocol_labels
                    )
                    self.logger.info(f"完成 {model_type} 模型的增量训练")
                    return model, metrics
                except Exception as e:
                    self.logger.warning(f"增量训练失败，回退到全量训练: {str(e)}")
            
            # 全量训练
            model, metrics, _ = self.base_trainer.train_new_model(
                model_type=model_type,
                X=X,
                y=y,
                protocol_labels=protocol_labels
            )
            self.logger.info(f"完成 {model_type} 模型的全量训练")
            return model, metrics
            
        except Exception as e:
            self.logger.error(f"训练 {model_type} 模型时出错: {str(e)}")
            raise
    
    def _perform_optimization(
        self,
        model,
        model_type: str,
        metrics: Dict[str, float],
        protocol: Optional[int]
    ) -> None:
        """执行优化"""
        try:
            self.logger.info(f"开始对 {model_type} 模型进行优化")
            
            # 获取特征重要性
            feature_importance = None
            if hasattr(model, "get_feature_importance"):
                try:
                    feature_importance = model.get_feature_importance()
                except Exception as e:
                    self.logger.warning(f"获取特征重要性时出错: {str(e)}")
            
            # 基于评估结果进行优化
            optimization_result = self.feedback_optimizer.optimize_based_on_evaluation(
                model_type=model_type,
                evaluation_metrics=metrics,
                protocol=protocol,
                feature_importance=feature_importance,
                model_factory=self.model_factory
            )
            
            # 记录优化建议
            if optimization_result.get("recommendations"):
                self.logger.info(
                    f"{model_type} 模型优化建议: "
                    f"{'; '.join(optimization_result['recommendations'])}"
                )
            
            # 保存优化历史
            self.feedback_optimizer.save_optimization_history()
            
        except Exception as e:
            self.logger.error(f"优化过程出错: {str(e)}", exc_info=True)
    
    def _tune_hyperparameters(
        self,
        current_model_type: str,
        metrics: Dict[str, float],
        iteration: int
    ) -> str:
        """调整超参数（简化版实现）"""
        # 这里可以实现更复杂的超参数调优逻辑
        # 简化起见，我们只在特定条件下切换模型类型
        current_f1 = metrics.get("f1", 0)
        
        # 如果性能不佳且不是LSTM，尝试切换到LSTM
        if current_f1 < 0.7 and current_model_type != "lstm" and iteration <= 3:
            self.logger.info("性能不佳，尝试切换到LSTM模型")
            return "lstm"
        
        # 保持当前模型类型
        return current_model_type
    
    def _save_best_model(self, model, model_type: str, metrics: Dict[str, float], iteration: int):
        """保存最佳模型"""
        try:
            model_path = self.model_factory.save_model(model, model_type)
            self.logger.info(f"保存第 {iteration} 轮的最佳 {model_type} 模型到: {model_path}")
        except Exception as e:
            self.logger.error(f"保存最佳模型时出错: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取AutoML训练器状态"""
        return {
            "is_running": self._is_running,
            "iterations_completed": len(self.iteration_history),
            "best_model": self.best_model_info,
            "config": self.automl_config
        }
    
    def get_iteration_history(self) -> List[Dict]:
        """获取迭代历史"""
        return self.iteration_history.copy()