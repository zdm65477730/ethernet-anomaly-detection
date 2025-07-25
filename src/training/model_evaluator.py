import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from src.utils.logger import get_logger
from src.config.config_manager import ConfigManager
from src.features.protocol_specs import get_protocol_spec, get_protocol_name

class ModelEvaluator:
    """模型评估器，计算各种评估指标并根据结果触发特征或策略优化"""
    
    def __init__(self, config=None):
        self.config = config or ConfigManager()
        self.logger = get_logger("training.evaluator")
        
        # 评估报告保存目录
        self.report_dir = self.config.get("training.report_dir", "reports/evaluations")
        import os
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
            except Exception as e:
                self.logger.warning(f"获取特征重要性失败: {str(e)}")
        
        # 生成评估报告
        report_path = self._generate_report(
            model_type=model_type,
            overall_metrics=overall_metrics,
            protocol_metrics=protocol_metrics,
            feature_importance=feature_importance,
            feature_names=feature_names,
            output_path=output_path
        )
        
        # 检查性能是否低于阈值
        self._check_performance_thresholds(model_type, overall_metrics, protocol_metrics)
        
        self.logger.info(
            f"模型评估完成，耗时 {time.time() - start_time:.2f}秒，"
            f"总体F1: {overall_metrics['f1']:.4f}"
        )
        
        return overall_metrics, report_path
    
    def _calculate_overall_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """计算总体评估指标"""
        metrics = {
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "support": len(y_true),
            "positive_rate": np.mean(y_true)  # 正样本比例（异常比例）
        }
        
        # 计算混淆矩阵元素
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tpr": tp / (tp + fn) if (tp + fn) > 0 else 0,  # 真阳性率
            "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0   # 假阳性率
        })
        
        # 计算AUC（如果有概率预测）
        if y_pred_proba is not None:
            try:
                metrics["auc"] = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                metrics["auc"] = 0.0
                self.logger.warning("计算AUC失败，可能是因为标签中只有一个类别")
        
        return metrics
    
    def _evaluate_by_protocol(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray],
        protocol_labels: List[int]
    ) -> Dict[str, Dict[str, float]]:
        """按协议评估模型性能"""
        protocol_metrics = {}
        unique_protocols = np.unique(protocol_labels)
        
        for proto in unique_protocols:
            # 筛选该协议的样本
            mask = np.array(protocol_labels) == proto
            y_true_proto = y_true[mask]
            y_pred_proto = y_pred[mask]
            
            # 跳过样本数不足的协议
            if len(y_true_proto) < self.min_samples_per_protocol:
                self.logger.debug(
                    f"协议 {get_protocol_name(proto)} 样本数不足 "
                    f"({len(y_true_proto)} < {self.min_samples_per_protocol})，跳过评估"
                )
                continue
            
            # 计算该协议的评估指标
            metrics = {
                "precision": precision_score(y_true_proto, y_pred_proto),
                "recall": recall_score(y_true_proto, y_pred_proto),
                "f1": f1_score(y_true_proto, y_pred_proto),
                "support": len(y_true_proto),
                "positive_rate": np.mean(y_true_proto)
            }
            
            # 混淆矩阵
            tn, fp, fn, tp = confusion_matrix(y_true_proto, y_pred_proto).ravel()
            metrics.update({
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "tpr": tp / (tp + fn) if (tp + fn) > 0 else 0,
                "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0
            })
            
            # AUC
            if y_pred_proba is not None:
                try:
                    metrics["auc"] = roc_auc_score(y_true_proto, y_pred_proba[mask])
                except ValueError:
                    metrics["auc"] = 0.0
            
            # 保存协议指标
            protocol_name = get_protocol_name(proto)
            protocol_metrics[protocol_name] = metrics
            self.logger.debug(
                f"协议 {protocol_name} 评估: F1={metrics['f1']:.4f}, "
                f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}"
            )
        
        return protocol_metrics
    
    def _generate_report(
        self,
        model_type: str,
        overall_metrics: Dict[str, float],
        protocol_metrics: Dict[str, Dict[str, float]],
        feature_importance: Optional[Dict[str, float]],
        feature_names: Optional[List[str]],
        output_path: Optional[str] = None
    ) -> str:
        """生成详细评估报告"""
        import os
        from datetime import datetime
        
        # 生成报告路径
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.report_dir, f"{model_type}_eval_{timestamp}.txt")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 写入报告
        with open(output_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write(f"模型评估报告 - {model_type}\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            # 总体指标
            f.write("1. 总体性能指标\n")
            f.write("-" * 40 + "\n")
            f.write(f"样本总数: {overall_metrics['support']:,}\n")
            f.write(f"异常样本比例: {overall_metrics['positive_rate']:.2%}\n\n")
            f.write(f"准确率 (Precision): {overall_metrics['precision']:.4f}\n")
            f.write(f"召回率 (Recall): {overall_metrics['recall']:.4f}\n")
            f.write(f"F1分数: {overall_metrics['f1']:.4f}\n")
            if "auc" in overall_metrics:
                f.write(f"AUC: {overall_metrics['auc']:.4f}\n")
            f.write("\n")
            f.write("混淆矩阵:\n")
            f.write(f"  真阳性 (TP): {overall_metrics['tp']:,}\n")
            f.write(f"  真阴性 (TN): {overall_metrics['tn']:,}\n")
            f.write(f"  假阳性 (FP): {overall_metrics['fp']:,}\n")
            f.write(f"  假阴性 (FN): {overall_metrics['fn']:,}\n")
            f.write(f"  真阳性率 (TPR): {overall_metrics['tpr']:.4f}\n")
            f.write(f"  假阳性率 (FPR): {overall_metrics['fpr']:.4f}\n\n")
            
            # 按协议指标
            if protocol_metrics:
                f.write("2. 按协议性能指标\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'协议':<10} {'F1分数':<10} {'准确率':<10} {'召回率':<10} {'样本数':<10}\n")
                f.write("-" * 60 + "\n")
                
                # 按F1排序
                sorted_protocols = sorted(
                    protocol_metrics.items(),
                    key=lambda x: x[1]["f1"],
                    reverse=True
                )
                
                for proto_name, metrics in sorted_protocols:
                    f.write(
                        f"{proto_name:<10} {metrics['f1']:.4f}     "
                        f"{metrics['precision']:.4f}     {metrics['recall']:.4f}     "
                        f"{metrics['support']:<10}\n"
                    )
                f.write("\n")
            
            # 特征重要性
            if feature_importance:
                f.write("3. 特征重要性\n")
                f.write("-" * 40 + "\n")
                
                # 按重要性排序
                sorted_features = sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                for i, (feature, importance) in enumerate(sorted_features[:20]):  # 只显示前20个
                    f.write(f"{i+1}. {feature:<30} {importance:.6f}\n")
                
                if len(sorted_features) > 20:
                    f.write(f"... 还有 {len(sorted_features) - 20} 个特征未显示\n")
                f.write("\n")
        
        self.logger.info(f"评估报告已保存至: {output_path}")
        return output_path
    
    def _check_performance_thresholds(
        self,
        model_type: str,
        overall_metrics: Dict[str, float],
        protocol_metrics: Dict[str, Dict[str, float]]
    ) -> None:
        """检查性能是否低于阈值，低于则发出警告"""
        # 检查总体性能
        for metric, threshold in self.performance_thresholds.items():
            if overall_metrics.get(metric, 0) < threshold:
                self.logger.warning(
                    f"模型 {model_type} 总体{metric}低于阈值 "
                    f"({overall_metrics[metric]:.4f} < {threshold})"
                )
        
        # 检查各协议性能
        for proto_name, metrics in protocol_metrics.items():
            for metric, threshold in self.performance_thresholds.items():
                if metrics.get(metric, 0) < threshold:
                    self.logger.warning(
                        f"协议 {proto_name} 在{metric}上低于阈值 "
                        f"({metrics[metric]:.4f} < {threshold})"
                    )
    
    def compare_models(
        self,
        model_metrics: Dict[str, Dict[str, float]],
        protocol_level: bool = False
    ) -> Dict[str, any]:
        """
        比较多个模型的性能
        
        参数:
            model_metrics: 模型指标字典，格式 {model_type: metrics_dict}
            protocol_level: 是否按协议级别比较
            
        返回:
            比较结果，包含最佳模型和各模型排名
        """
        # 按F1分数排序
        sorted_models = sorted(
            model_metrics.items(),
            key=lambda x: x[1]["f1"],
            reverse=True
        )
        
        comparison_result = {
            "best_model": sorted_models[0][0],
            "best_model_metrics": sorted_models[0][1],
            "model_ranking": [model[0] for model in sorted_models],
            "full_comparison": {}
        }
        
        # 构建详细比较表
        for model_type, metrics in sorted_models:
            comparison_result["full_comparison"][model_type] = {
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "auc": metrics.get("auc", 0),
                "support": metrics["support"]
            }
        
        self.logger.info(f"模型比较完成，最佳模型: {comparison_result['best_model']}")
        return comparison_result
