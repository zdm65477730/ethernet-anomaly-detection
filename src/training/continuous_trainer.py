import time
import os
import threading
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from src.utils.logger import get_logger
from src.config.config_manager import ConfigManager
from src.data.data_storage import DataStorage
from src.models.model_factory import ModelFactory
from src.models.model_selector import ModelSelector
from src.features.protocol_specs import get_all_protocols
from .model_trainer import ModelTrainer
from .incremental_trainer import IncrementalTrainer
from .model_evaluator import ModelEvaluator
from .feedback_optimizer import FeedbackOptimizer

class ContinuousTrainer:
    """持续训练器，实现自动检查新数据、训练模型并优化的循环"""
    
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
        
        self.logger = get_logger("training.continuous")
        
        # 持续训练配置
        self.continuous_config = {
            "check_interval": self.config.get("training.continuous.check_interval", 3600),  # 1小时
            "min_samples": self.config.get("training.continuous.min_samples", 1000),        # 最小新样本数
            "max_history_days": self.config.get("training.continuous.max_history_days", 30), # 最大历史数据天数
            "retrain_full_interval": self.config.get("training.continuous.retrain_full_interval", 86400 * 7),  # 7天
            "models_dir": self.config.get("model.models_dir", "models"),
            "protocol_model_map": self.config.get("training.continuous.protocol_model_map", {}),
            "enable_auto_optimization": self.config.get("training.continuous.enable_auto_optimization", True)
        }
        
        # 状态变量
        self._is_running = False
        self._thread = None
        self._last_check_time = 0
        self._last_full_retrain_time = {}  # 按模型类型记录上次全量训练时间
        self._new_data_count = {}          # 记录各协议的新数据计数
        
        # 创建必要目录
        os.makedirs(self.continuous_config["models_dir"], exist_ok=True)
    
    def start(
        self,
        check_interval: Optional[int] = None,
        min_samples: Optional[int] = None,
        max_history_days: Optional[int] = None
    ) -> None:
        """
        启动持续训练循环
        
        参数:
            check_interval: 检查新数据的时间间隔(秒)
            min_samples: 触发训练的最小新样本数
            max_history_days: 使用的最大历史数据天数
        """
        if self._is_running:
            self.logger.warning("持续训练已在运行中")
            return
            
        # 更新配置
        if check_interval is not None:
            self.continuous_config["check_interval"] = check_interval
        if min_samples is not None:
            self.continuous_config["min_samples"] = min_samples
        if max_history_days is not None:
            self.continuous_config["max_history_days"] = max_history_days
        
        self._is_running = True
        
        # 启动训练循环线程
        self._thread = threading.Thread(
            target=self._continuous_training_loop,
            daemon=True
        )
        self._thread.start()
        
        self.logger.info(
            f"持续训练已启动，检查间隔: {self.continuous_config['check_interval']}秒, "
            f"最小样本数: {self.continuous_config['min_samples']}"
        )
    
    def stop(self) -> None:
        """停止持续训练循环"""
        if not self._is_running:
            self.logger.warning("持续训练未在运行")
            return
            
        self._is_running = False
        
        # 等待线程结束
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                self.logger.warning("持续训练线程未能正常终止")
        
        self.logger.info("持续训练已停止")
    
    def _continuous_training_loop(self) -> None:
        """持续训练主循环"""
        self.logger.info("持续训练循环开始")
        
        try:
            while self._is_running:
                current_time = time.time()
                
                # 检查是否需要检查新数据
                if current_time - self._last_check_time >= self.continuous_config["check_interval"]:
                    self._last_check_time = current_time
                    self._check_and_train()
                
                # 等待一段时间或直到被唤醒
                sleep_time = max(1, self.continuous_config["check_interval"] - (time.time() - current_time))
                for _ in range(int(sleep_time)):
                    if not self._is_running:
                        break
                    time.sleep(1)
                
        except Exception as e:
            self.logger.error(f"持续训练循环出错: {str(e)}", exc_info=True)
            # 尝试恢复
            if self._is_running:
                self.logger.info("尝试恢复持续训练循环...")
                time.sleep(60)  # 等待一分钟再重试
                self._continuous_training_loop()
    
    def _check_and_train(self) -> None:
        """检查新数据并执行训练"""
        self.logger.info("开始检查新数据并训练模型")
        
        try:
            # 获取所有协议类型
            all_protocols = get_all_protocols()
            
            # 1. 检查通用模型是否需要训练（所有协议数据）
            self._check_and_train_for_protocol(None, "通用模型")
            
            # 2. 检查各协议专用模型是否需要训练
            for proto_num, proto_info in all_protocols.items():
                self._check_and_train_for_protocol(proto_num, proto_info["name"])
                
        except Exception as e:
            self.logger.error(f"检查和训练过程出错: {str(e)}", exc_info=True)
    
    def _check_and_train_for_protocol(self, protocol: Optional[int], protocol_name: str) -> None:
        """检查特定协议的新数据并训练模型"""
        # 确定数据查询时间范围
        time_threshold = self._get_data_time_threshold()
        
        # 查询新数据
        new_data_count = self.data_storage.count_new_data_since(
            timestamp=time_threshold,
            protocol=protocol
        )
        
        self.logger.info(
            f"{protocol_name} 新数据检查: 发现 {new_data_count} 个新样本 "
            f"(自 {time.ctime(time_threshold)})"
        )
        
        # 检查是否有足够的新数据
        if new_data_count < self.continuous_config["min_samples"]:
            # 更新新数据计数，但不触发训练
            self._new_data_count[protocol or "general"] = new_data_count
            return
        
        # 有足够的新数据，加载并训练
        try:
            # 加载历史数据和新数据
            X, y, protocol_labels, feature_names = self._load_training_data(
                protocol=protocol,
                time_threshold=time_threshold
            )
            
            self.logger.info(
                f"为 {protocol_name} 加载训练数据: 总样本数 {len(X)}, "
                f"异常比例 {np.mean(y):.2%}"
            )
            
            # 确定最佳模型类型
            if protocol is not None:
                # 协议专用模型
                model_type = self.model_selector.select_best_model(
                    protocol=protocol,
                    candidates=list(self.continuous_config["protocol_model_map"].values())
                )
            else:
                # 通用模型，使用配置的默认模型
                model_type = self.config.get("training.default_model", "xgboost")
            
            self.logger.info(f"{protocol_name} 选择模型类型: {model_type}")
            
            # 检查是否需要全量重训
            current_time = time.time()
            last_full_time = self._last_full_retrain_time.get(model_type, 0)
            need_full_retrain = (
                current_time - last_full_time >= self.continuous_config["retrain_full_interval"] or
                self.incremental_trainer.should_retrain_full(model_type)
            )
            
            if need_full_retrain:
                # 执行全量重训
                self.logger.info(f"{protocol_name} 触发全量重训 {model_type} 模型")
                model, metrics, _ = self.base_trainer.train_new_model(
                    model_type=model_type,
                    X=X,
                    y=y,
                    protocol_labels=protocol_labels
                )
                self._last_full_retrain_time[model_type] = current_time
                
                # 更新模型选择器的性能记录
                self.model_selector.update_performance(
                    protocol=protocol or "general",
                    model_type=model_type,
                    metrics=metrics
                )
                
            else:
                # 尝试加载现有模型进行增量训练
                try:
                    model = self.model_factory.load_latest_model(model_type)
                    self.logger.info(f"为 {protocol_name} 加载现有 {model_type} 模型进行增量训练")
                    
                    # 执行增量训练
                    model, metrics, triggered_retrain = self.incremental_trainer.update_model_incrementally(
                        model_type=model_type,
                        new_X=X,
                        new_y=y,
                        model=model,
                        protocol_labels=protocol_labels
                    )
                    
                    # 如果触发了全量重训，更新时间戳
                    if triggered_retrain:
                        self._last_full_retrain_time[model_type] = current_time
                    
                    # 更新模型选择器的性能记录
                    self.model_selector.update_performance(
                        protocol=protocol or "general",
                        model_type=model_type,
                        metrics=metrics
                    )
                    
                except Exception as e:
                    self.logger.error(f"加载或增量训练 {model_type} 模型失败: {str(e)}", exc_info=True)
                    # 回退到全量训练
                    self.logger.info(f"回退到全量训练 {model_type} 模型")
                    model, metrics, _ = self.base_trainer.train_new_model(
                        model_type=model_type,
                        X=X,
                        y=y,
                        protocol_labels=protocol_labels
                    )
                    self._last_full_retrain_time[model_type] = current_time
                    
                    # 更新模型选择器的性能记录
                    self.model_selector.update_performance(
                        protocol=protocol or "general",
                        model_type=model_type,
                        metrics=metrics
                    )
            
            # 保存训练好的模型
            model_path = self.model_factory.save_model(model, model_type)
            self.logger.info(f"{protocol_name} 的 {model_type} 模型已保存到: {model_path}")
            
        except Exception as e:
            self.logger.error(f"为 {protocol_name} 训练模型失败: {str(e)}", exc_info=True)
    
    def _get_data_time_threshold(self) -> float:
        """获取数据时间阈值，用于确定新数据范围"""
        max_history_seconds = self.continuous_config["max_history_days"] * 24 * 3600
        return time.time() - max_history_seconds
    
    def _perform_auto_optimization(
        self, 
        model, 
        model_type: str, 
        metrics: Dict[str, float], 
        protocol: Optional[int], 
        protocol_name: str
    ):
        """执行自动优化"""
        try:
            self.logger.info(f"开始对 {protocol_name} 的 {model_type} 模型进行自动优化")
            
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
                    f"{protocol_name} {model_type} 模型优化建议: "
                    f"{'; '.join(optimization_result['recommendations'])}"
                )
            
            # 保存优化历史
            self.feedback_optimizer.save_optimization_history()
            
        except Exception as e:
            self.logger.error(f"自动优化过程出错: {str(e)}", exc_info=True)
    
    def _load_training_data(
        self, 
        protocol: Optional[int], 
        time_threshold: float
    ) -> Tuple[np.ndarray, np.ndarray, Optional[List[int]], List[str]]:
        """加载训练数据"""
        # 这里应该实现实际的数据加载逻辑
        # 为简化起见，我们返回空的数据结构
        # 实际实现应该从data_storage加载数据
        return np.array([]), np.array([]), None, []
    
    def get_status(self) -> Dict[str, any]:
        """获取持续训练器状态"""
        return {
            "is_running": self._is_running,
            "last_check_time": self._last_check_time,
            "last_full_retrain_time": self._last_full_retrain_time,
            "new_data_count": self._new_data_count,
            "config": self.continuous_config
        }
        
    def trigger_manual_training(
        self, 
        model_type: str, 
        protocol: Optional[int] = None
    ) -> Tuple[bool, str]:
        """
        手动触发模型训练
            
        参数:
            model_type: 模型类型
            protocol: 协议编号（None表示通用模型）
                
        返回:
            (训练是否成功, 结果信息)
        """
        try:
            protocol_name = "通用模型"
            if protocol is not None:
                proto_spec = get_all_protocols().get(protocol)
                protocol_name = proto_spec["name"] if proto_spec else f"协议_{protocol}"
                
            self.logger.info(f"手动触发 {protocol_name} 的 {model_type} 模型训练")
            self._check_and_train_for_protocol(protocol, protocol_name)
            return True, f"{protocol_name} 的 {model_type} 模型训练完成"
        except Exception as e:
            error_msg = f"手动训练失败: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, error_msg