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
from .model_trainer import ModelTrainer
from .incremental_trainer import IncrementalTrainer
from .model_evaluator import ModelEvaluator

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
            "protocol_model_map": self.config.get("training.continuous.protocol_model_map", {})
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
            from src.features.protocol_specs import get_all_protocols
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