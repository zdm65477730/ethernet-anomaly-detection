import time
import json
import time
import os
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from src.config.config_manager import ConfigManager
from src.utils.logger import get_logger
from src.features.stat_extractor import StatFeatureExtractor
from src.features.temporal_extractor import TemporalFeatureExtractor
from .model_trainer import ModelTrainer
from .incremental_trainer import IncrementalTrainer

class FeedbackOptimizer:
    """基于反馈进行模型优化的组件"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """初始化反馈优化器"""
        self.config = config or ConfigManager()
        self.logger = get_logger("training.feedback_optimizer")
        self.optimization_history: List[Dict] = []
        
        # 优化配置
        self.optimization_config = {
            "enable_feature_optimization": self.config.get("training.feedback.enable_feature_optimization", True),
            "enable_model_optimization": self.config.get("training.feedback.enable_model_optimization", True),
            "f1_threshold_for_model_change": self.config.get("training.feedback.f1_threshold_for_model_change", 0.3),
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
        self.optimization_history: List[Dict] = []
        
        # 特征性能记录
        self.feature_performance: Dict[str, Dict[str, float]] = {}
        
        self.logger.info("反馈优化器初始化完成")
    
    def optimize_based_on_evaluation(
        self,
        model_type: str,
        evaluation_metrics: Dict[str, Any],
        protocol: Optional[int],
        feature_importance: Optional[Dict[str, float]],
        model_factory: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        基于评估结果进行优化
        
        Args:
            model_type: 模型类型
            evaluation_metrics: 评估指标
            protocol: 协议类型
            feature_importance: 特征重要性
            model_factory: 模型工厂（可选）
            
        Returns:
            优化建议字典
        """
        self.logger.info(f"开始基于评估结果优化 {model_type} 模型")
        
        recommendations = {
            "model_type": model_type,
            "protocol": protocol,
            "changes": {},
            "reasoning": []
        }
        
        # 获取当前配置
        current_config = self.config.get(model_type, {})
        changes = recommendations["changes"]
        reasoning = recommendations["reasoning"]
        
        # 1. 分析F1分数并给出建议
        f1_score = evaluation_metrics.get("f1", 0)
        precision = evaluation_metrics.get("precision", 0)
        recall = evaluation_metrics.get("recall", 0)
        
        self.logger.info(f"当前模型性能 - F1: {f1_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        # 如果F1分数较低，考虑更换模型类型
        if f1_score < self.optimization_config["f1_threshold_for_model_change"]:
            # 为协议选择更好的模型类型
            if protocol is not None and model_factory is not None:
                from src.models.model_selector import ModelSelector
                selector = ModelSelector()
                better_model = selector.select_best_model(protocol=protocol)
                if better_model and better_model != model_type:
                    changes["model_type"] = better_model
                    reasoning.append(f"F1分数({f1_score:.4f})低于阈值({self.optimization_config['f1_threshold_for_model_change']})，建议更换为{better_model}模型")
        
        # 2. 分析精确率和召回率平衡
        if abs(precision - recall) > 0.1:  # 如果差异较大
            if precision > recall:
                reasoning.append("精确率高于召回率，模型偏向保守预测")
            else:
                reasoning.append("召回率高于精确率，模型偏向激进预测")
        
        # 3. 如果提供了特征重要性，分析特征
        if feature_importance:
            # 找出最重要的特征
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = [f[0] for f in sorted_features[:5]]
            reasoning.append(f"最重要的5个特征: {', '.join(top_features)}")
            
            # 检查是否有特征重要性接近0
            low_importance_features = [f[0] for f in sorted_features if f[1] < 0.01]
            if len(low_importance_features) > len(sorted_features) * 0.5:
                reasoning.append(f"超过50%的特征重要性低于0.01，考虑特征选择")
                changes["feature_selection"] = {
                    "keep_features": top_features,
                    "remove_features": low_importance_features
                }
        
        # 4. 根据性能调整模型参数
        if f1_score < 0.3:
            # 性能较差，建议增加模型复杂度
            if model_type == "mlp":
                hidden_units = current_config.get("hidden_units", [64, 32])
                # 增加隐藏单元数量
                new_hidden_units = [units * 2 for units in hidden_units]
                changes["hidden_units"] = new_hidden_units
                reasoning.append(f"增加MLP隐藏单元数量: {hidden_units} -> {new_hidden_units}")
                
            elif model_type == "lstm":
                lstm_units = current_config.get("lstm_units", [32, 16])
                new_lstm_units = [units * 2 for units in lstm_units]
                changes["lstm_units"] = new_lstm_units
                reasoning.append(f"增加LSTM单元数量: {lstm_units} -> {new_lstm_units}")
        
        elif f1_score > 0.7:
            # 性能较好，可以尝试减少模型复杂度防止过拟合
            if model_type == "mlp":
                hidden_units = current_config.get("hidden_units", [128, 64, 32])
                if len(hidden_units) > 2:
                    # 减少层数
                    new_hidden_units = hidden_units[:-1]
                    changes["hidden_units"] = new_hidden_units
                    reasoning.append(f"减少MLP层数以防止过拟合: {hidden_units} -> {new_hidden_units}")
        
        # 5. 根据精确率和召回率的平衡调整阈值
        if abs(precision - recall) > 0.1:
            current_threshold = current_config.get("threshold", 0.5)
            if precision > recall:
                # 精确率高，召回率低，降低阈值以提高召回率
                new_threshold = max(0.1, current_threshold - 0.05)
                changes["threshold"] = new_threshold
                reasoning.append(f"降低分类阈值以提高召回率: {current_threshold} -> {new_threshold}")
            else:
                # 召回率高，精确率低，提高阈值以提高精确率
                new_threshold = min(0.9, current_threshold + 0.05)
                changes["threshold"] = new_threshold
                reasoning.append(f"提高分类阈值以提高精确率: {current_threshold} -> {new_threshold}")
        
        # 6. 根据F1分数考虑使用集成模型
        if f1_score < 0.4 and model_type in ["xgboost", "random_forest", "mlp", "lstm"]:
            changes["model_type"] = "ensemble"
            reasoning.append(f"F1分数({f1_score:.4f})较低，建议使用集成模型提高性能")
        
        self.logger.info(f"优化建议: {len(reasoning)} 条")
        for i, reason in enumerate(reasoning, 1):
            self.logger.info(f"  {i}. {reason}")
        
        # 记录优化历史
        self.optimization_history.append({
            "timestamp": time.time(),
            "model_type": model_type,
            "protocol": protocol,
            "metrics": evaluation_metrics,
            "recommendations": recommendations
        })
        
        return recommendations

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
        serializable_history = []
        for item in self.optimization_history:
            serializable_history.append(self._make_serializable(item))
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"优化历史已保存至: {filepath}")
    
    def _make_serializable(self, obj):
        """将对象转换为可JSON序列化的格式"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()  # 转换numpy类型为Python原生类型
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
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