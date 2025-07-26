import time
import json
import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from src.utils.logger import get_logger
from src.config.config_manager import ConfigManager
from src.features.stat_extractor import StatFeatureExtractor
from src.features.temporal_extractor import TemporalFeatureExtractor
from src.features.protocol_specs import get_all_protocols
from .model_evaluator import ModelEvaluator
from .model_trainer import ModelTrainer
from .incremental_trainer import IncrementalTrainer

class FeedbackOptimizer:
    """反馈优化器，根据模型评估结果优化特征工程和模型训练"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        self.config = config or ConfigManager()
        self.logger = get_logger("training.feedback_optimizer")
        
        # 优化配置
        self.optimization_config = {
            "enable_feature_optimization": self.config.get("training.feedback.enable_feature_optimization", True),
            "enable_model_optimization": self.config.get("training.feedback.enable_model_optimization", True),
            "performance_threshold": self.config.get("training.feedback.performance_threshold", 0.7),
            "improvement_threshold": self.config.get("training.feedback.improvement_threshold", 0.05),
            "max_optimization_rounds": self.config.get("training.feedback.max_optimization_rounds", 3),
            "min_samples_for_optimization": self.config.get("training.feedback.min_samples_for_optimization", 1000)
        }
        
        # 特征提取器
        self.stat_extractor = StatFeatureExtractor()
        self.temporal_extractor = TemporalFeatureExtractor()
        
        # 训练器
        self.model_trainer = ModelTrainer(None, self.config)  # model_factory将在使用时传入
        self.incremental_trainer = IncrementalTrainer(None, self.config)  # model_factory将在使用时传入
        
        # 优化历史记录
        self.optimization_history: Dict[str, List[Dict]] = {}
        
        # 特征性能记录
        self.feature_performance: Dict[str, Dict[str, float]] = {}
        
        self.logger.info("反馈优化器初始化完成")
    
    def optimize_based_on_evaluation(
        self,
        model_type: str,
        evaluation_metrics: Dict[str, float],
        protocol: Optional[int] = None,
        feature_importance: Optional[Dict[str, float]] = None,
        model_factory=None
    ) -> Dict[str, Any]:
        """
        根据模型评估结果进行优化
        
        参数:
            model_type: 模型类型
            evaluation_metrics: 评估指标
            protocol: 协议编号
            feature_importance: 特征重要性
            model_factory: 模型工厂
            
        返回:
            优化建议和执行结果
        """
        self.logger.info(f"开始基于评估结果优化 {model_type} 模型")
        
        if model_factory:
            self.model_trainer.model_factory = model_factory
            self.incremental_trainer.model_factory = model_factory
        
        optimization_result = {
            "timestamp": time.time(),
            "model_type": model_type,
            "protocol": protocol,
            "metrics": evaluation_metrics,
            "optimizations": [],
            "recommendations": []
        }
        
        # 1. 检查模型性能是否需要优化
        performance_check = self._check_model_performance(evaluation_metrics)
        if performance_check["needs_optimization"]:
            optimization_result["optimizations"].append(performance_check)
            optimization_result["recommendations"].extend(performance_check["recommendations"])
        
        # 2. 根据特征重要性优化特征工程
        if self.optimization_config["enable_feature_optimization"] and feature_importance:
            feature_optimization = self._optimize_features(feature_importance, protocol)
            if feature_optimization["changes_made"]:
                optimization_result["optimizations"].append(feature_optimization)
                optimization_result["recommendations"].extend(feature_optimization["recommendations"])
                self._update_feature_performance(feature_importance)
        
        # 3. 根据性能调整模型参数
        if self.optimization_config["enable_model_optimization"]:
            model_optimization = self._optimize_model_parameters(model_type, evaluation_metrics)
            if model_optimization["changes_made"]:
                optimization_result["optimizations"].append(model_optimization)
                optimization_result["recommendations"].extend(model_optimization["recommendations"])
        
        # 记录优化历史
        protocol_key = str(protocol) if protocol is not None else "general"
        if protocol_key not in self.optimization_history:
            self.optimization_history[protocol_key] = []
        self.optimization_history[protocol_key].append(optimization_result)
        
        self.logger.info(f"{model_type} 模型优化完成")
        return optimization_result
    
    def _check_model_performance(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """检查模型性能是否需要优化"""
        performance_check = {
            "type": "performance_check",
            "needs_optimization": False,
            "recommendations": [],
            "details": {}
        }
        
        # 检查主要指标是否低于阈值
        main_metrics = ["f1", "precision", "recall"]
        low_performance_metrics = []
        
        for metric in main_metrics:
            if metric in metrics and metrics[metric] < self.optimization_config["performance_threshold"]:
                low_performance_metrics.append((metric, metrics[metric]))
                performance_check["needs_optimization"] = True
        
        if low_performance_metrics:
            performance_check["details"]["low_performance"] = low_performance_metrics
            performance_check["recommendations"].append(
                "模型性能低于阈值，建议重新训练或调整特征工程"
            )
            
            # 如果性能过低，建议更激进的优化
            very_low_metrics = [
                metric for metric, value in low_performance_metrics 
                if value < self.optimization_config["performance_threshold"] * 0.7
            ]
            if very_low_metrics:
                performance_check["recommendations"].append(
                    f"以下指标性能极低: {', '.join(very_low_metrics)}，建议全量重训"
                )
        
        return performance_check
    
    def _optimize_features(self, feature_importance: Dict[str, float], protocol: Optional[int]) -> Dict[str, Any]:
        """根据特征重要性优化特征工程"""
        feature_optimization = {
            "type": "feature_optimization",
            "changes_made": False,
            "recommendations": [],
            "details": {}
        }
        
        # 分析特征重要性
        if not feature_importance:
            return feature_optimization
        
        # 按重要性排序
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # 识别低重要性特征
        low_importance_features = [
            feature for feature, importance in sorted_features 
            if importance < 0.01  # 重要性低于1%的特征
        ]
        
        # 识别高重要性特征
        high_importance_features = [
            feature for feature, importance in sorted_features 
            if importance > 0.1  # 重要性高于10%的特征
        ]
        
        feature_optimization["details"]["low_importance_features"] = low_importance_features
        feature_optimization["details"]["high_importance_features"] = high_importance_features
        
        # 如果有低重要性特征，建议移除
        if low_importance_features:
            feature_optimization["changes_made"] = True
            feature_optimization["recommendations"].append(
                f"发现 {len(low_importance_features)} 个低重要性特征，建议在下次训练中考虑移除"
            )
        
        # 如果有高重要性特征，建议加强相关特征工程
        if high_importance_features:
            feature_optimization["changes_made"] = True
            feature_optimization["recommendations"].append(
                f"发现 {len(high_importance_features)} 个高重要性特征，建议加强相关特征工程"
            )
        
        return feature_optimization
    
    def _optimize_model_parameters(self, model_type: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        """根据性能调整模型参数"""
        model_optimization = {
            "type": "model_optimization",
            "changes_made": False,
            "recommendations": [],
            "details": {}
        }
        
        # 基于性能调整参数的简单策略
        current_f1 = metrics.get("f1", 0)
        
        if current_f1 < self.optimization_config["performance_threshold"]:
            model_optimization["changes_made"] = True
            model_optimization["details"]["current_f1"] = current_f1
            
            # 根据模型类型提供不同的优化建议
            if model_type in ["xgboost", "random_forest"]:
                model_optimization["recommendations"].append(
                    "建议增加模型复杂度：增大n_estimators或max_depth"
                )
            elif model_type == "lstm":
                model_optimization["recommendations"].append(
                    "建议调整LSTM参数：增加隐藏层维度或训练轮数"
                )
            elif model_type == "mlp":
                model_optimization["recommendations"].append(
                    "建议调整MLP参数：增加层数或神经元数量"
                )
            else:
                model_optimization["recommendations"].append(
                    "建议调整模型超参数以提高性能"
                )
        
        return model_optimization
    
    def _update_feature_performance(self, feature_importance: Dict[str, float]):
        """更新特征性能记录"""
        for feature, importance in feature_importance.items():
            if feature not in self.feature_performance:
                self.feature_performance[feature] = {
                    "total_importance": 0,
                    "count": 0,
                    "average_importance": 0
                }
            
            self.feature_performance[feature]["total_importance"] += importance
            self.feature_performance[feature]["count"] += 1
            self.feature_performance[feature]["average_importance"] = (
                self.feature_performance[feature]["total_importance"] / 
                self.feature_performance[feature]["count"]
            )
    
    def get_feature_ranking(self) -> List[Tuple[str, float]]:
        """获取特征重要性排名"""
        ranking = [
            (feature, data["average_importance"]) 
            for feature, data in self.feature_performance.items()
        ]
        return sorted(ranking, key=lambda x: x[1], reverse=True)
    
    def suggest_feature_engineering(self) -> List[str]:
        """基于历史性能建议特征工程优化"""
        recommendations = []
        
        # 获取特征排名
        feature_ranking = self.get_feature_ranking()
        
        if not feature_ranking:
            return recommendations
        
        # 识别需要改进的特征
        low_performing_features = [
            feature for feature, avg_importance in feature_ranking
            if avg_importance < 0.01
        ]
        
        high_performing_features = [
            feature for feature, avg_importance in feature_ranking
            if avg_importance > 0.1
        ]
        
        if low_performing_features:
            recommendations.append(
                f"考虑移除或重构以下低价值特征: {', '.join(low_performing_features[:5])}"
            )
        
        if high_performing_features:
            recommendations.append(
                f"加强以下高价值特征的工程: {', '.join(high_performing_features[:3])}"
            )
        
        return recommendations
    
    def auto_adjust_feature_extractors(self):
        """自动调整特征提取器配置"""
        # 这里可以实现根据特征重要性动态调整特征提取器的逻辑
        # 例如启用/禁用某些特征，调整特征计算参数等
        pass
    
    def save_optimization_history(self, filepath: Optional[str] = None):
        """保存优化历史记录"""
        if filepath is None:
            reports_dir = self.config.get("training.report_dir", "reports/evaluations")
            filepath = os.path.join(reports_dir, "optimization_history.json")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 转换不可序列化的对象
        serializable_history = {}
        for key, value in self.optimization_history.items():
            serializable_history[key] = value
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"优化历史已保存至: {filepath}")
    
    def load_optimization_history(self, filepath: Optional[str] = None):
        """加载优化历史记录"""
        if filepath is None:
            reports_dir = self.config.get("training.report_dir", "reports/evaluations")
            filepath = os.path.join(reports_dir, "optimization_history.json")
        
        if not os.path.exists(filepath):
            self.logger.warning(f"优化历史文件不存在: {filepath}")
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.optimization_history = json.load(f)
            self.logger.info(f"优化历史已从 {filepath} 加载")
        except Exception as e:
            self.logger.error(f"加载优化历史失败: {str(e)}")