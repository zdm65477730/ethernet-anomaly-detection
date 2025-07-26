import time    def _estimate_feature_importance(
        self, 
        model, 
        X: np.ndarray, 
        y: np.ndarray, 
        feature_names: List[str]
    ) -> Dict[str, float]:
        """估计特征重要性（当模型不直接提供时）"""
        try:
            # 使用排列重要性方法估算
            from sklearn.inspection import permutation_importance
            
            # 计算基线性能
            baseline_score = model.score(X, y)
            
            # 计算每个特征的重要性
            importances = {}
            for i, feature_name in enumerate(feature_names):
                # 创建扰动数据
                X_permuted = X.copy()
                # 扰动第i个特征
                np.random.shuffle(X_permuted[:, i])
                
                # 计算扰动后的性能
                permuted_score = model.score(X_permuted, y)
                
                # 计算重要性（性能下降越大，特征越重要）
                importances[feature_name] = baseline_score - permuted_score
            
            # 归一化重要性
            total_importance = sum(abs(v) for v in importances.values())
            if total_importance > 0:
                importances = {k: abs(v) / total_importance for k, v in importances.items()}
            
            return importances
        except Exception as e:
            self.logger.warning(f"估算特征重要性时出错: {str(e)}")
            return None
    
    def _check_performance_and_recommend(self, metrics: Dict[str, float], model_type: str):
        """检查性能并给出优化建议"""
        recommendations = []
        
        # 检查各项指标是否低于阈值
        for metric, threshold in self.performance_thresholds.items():
            if metric in metrics and metrics[metric] < threshold:
                recommendations.append(
                    f"{metric.upper()} ({metrics[metric]:.3f}) 低于阈值 ({threshold})，"
                    f"建议优化模型或特征工程"
                )
        
        if recommendations:
            self.logger.warning(
                f"{model_type} 模型性能需要优化: {'; '.join(recommendations)}"
            )
            self.logger.info(
                "建议使用 feedback_optimizer 模块进行自动优化，"
                "或手动调整特征工程和模型参数"
            )
import json
import os
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from src.utils.logger import get_logger
from src.config.config_manager import ConfigManager
from src.features.protocol_specs import get_protocol_spec

class ModelEvaluator:
    """模型评估器，提供全面的模型性能评估功能"""
    
    def __init__(self, config=None):
        self.config = config or ConfigManager()
        self.logger = get_logger("training.evaluator")
        
        # 评估报告保存目录
        self.report_dir = self.config.get("training.report_dir", "reports/evaluations")
        os.makedirs(self.report_dir, exist_ok=True)
        
        # 最小样本数阈值（低于此值不进行协议级评估）
        self.min_samples_per_protocol = self.config.get("training.min_samples_per_protocol", 50)
        
        # 性能警告阈值
        self.performance_thresholds = {
            "f1": 0.7,
            "precision": 0.6,
            "recall": 0.6
        }

    def evaluate_model(
        self,
        model,
        model_type: str,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        protocol_labels: Optional[List[int]] = None,
        feature_names: Optional[List[str]] = None,
        output_path: Optional[str] = None
    ) -> Tuple[Dict[str, float], str]:
        """
        评估模型性能，生成评估报告
        
        参数:
            model: 已训练的模型
            model_type: 模型类型名称
            X_test: 测试特征数据
            y_test: 测试标签
            protocol_labels: 每个样本对应的协议标签列表
            feature_names: 特征名称列表
            output_path: 评估报告输出路径，为None则自动生成
            
        返回:
            总体评估指标字典和报告保存路径
        """
        start_time = time.time()
        
        # 确保输入数据格式正确
        if isinstance(X_test, pd.DataFrame):
            X_test_np = X_test.values
            if feature_names is None:
                feature_names = X_test.columns.tolist()
        else:
            X_test_np = X_test
        
        if isinstance(y_test, pd.Series):
            y_test_np = y_test.values
        else:
            y_test_np = y_test
        
        # 预测
        y_pred = model.predict(X_test_np)
        try:
            y_pred_proba = model.predict_proba(X_test_np)[:, 1]  # 异常概率
        except NotImplementedError:
            y_pred_proba = None
            self.logger.warning(f"{model_type} 不支持概率预测，无法计算AUC")
        
        # 计算总体评估指标
        overall_metrics = self._calculate_overall_metrics(y_test_np, y_pred, y_pred_proba)
        
        # 按协议评估（如果提供了协议标签）
        protocol_metrics = {}
        if protocol_labels is not None:
            protocol_metrics = self._evaluate_by_protocol(
                X_test_np, y_test_np, y_pred, y_pred_proba, protocol_labels
            )
        
        # 特征重要性分析（如果模型支持）
        feature_importance = None
        if hasattr(model, "get_feature_importance"):
            try:
                feature_importance = model.get_feature_importance()
                if feature_importance is None and feature_names:
                    # 如果模型没有返回特征重要性，尝试使用默认方法
                    feature_importance = self._estimate_feature_importance(
                        model, X_test_np, y_test_np, feature_names
                    )
            except Exception as e:
                self.logger.warning(f"获取特征重要性时出错: {str(e)}")
        
        # 生成评估报告
        report_data = {
            "model_type": model_type,
            "evaluation_time": time.time(),
            "sample_count": len(y_test_np),
            "anomaly_ratio": float(np.mean(y_test_np)),
            "overall_metrics": overall_metrics,
            "protocol_metrics": protocol_metrics,
            "feature_importance": feature_importance,
            "confusion_matrix": confusion_matrix(y_test_np, y_pred).tolist()
        }
        
        # 保存评估报告
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.report_dir, 
                f"{model_type}_evaluation_{timestamp}.json"
            )
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"模型评估完成，报告已保存至: {output_path}")
        
        # 检查性能是否需要优化
        self._check_performance_and_recommend(overall_metrics, model_type)
        
        return overall_metrics, output_path

    def _calculate_overall_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """计算总体评估指标"""
        metrics = {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_pred_proba is not None:
            try:
                metrics["auc"] = roc_auc_score(y_true, y_pred_proba)
            except Exception:
                metrics["auc"] = 0.0
                self.logger.warning("AUC计算失败")
        
        # 添加更多指标
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics.update({
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "accuracy": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        })
        
        return metrics

    def _evaluate_by_protocol(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray],
        protocol_labels: List[int]
    ) -> Dict[str, Dict[str, float]]:
        """按协议类型评估模型性能"""
        protocol_metrics = {}
        
        # 获取唯一协议类型
        unique_protocols = np.unique(protocol_labels)
        
        for protocol in unique_protocols:
            # 筛选该协议的样本
            mask = np.array(protocol_labels) == protocol
            if np.sum(mask) < self.min_samples_per_protocol:
                continue  # 样本数太少，跳过评估
            
            # 获取协议数据
            X_proto = X_test[mask]
            y_true_proto = y_test[mask]
            y_pred_proto = y_pred[mask]
            y_proba_proto = y_pred_proba[mask] if y_pred_proba is not None else None
            
            # 计算指标
            proto_metrics = self._calculate_overall_metrics(
                y_true_proto, y_pred_proto, y_proba_proto
            )