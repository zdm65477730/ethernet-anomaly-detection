import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """模型评估器，用于评估和比较不同模型的性能"""
    
    def __init__(self):
        """初始化模型评估器"""
        pass
    
    def evaluate_model(
        self, 
        model, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """
        全面评估模型性能
        
        Args:
            model: 训练好的模型
            X_test: 测试特征数据
            y_test: 测试标签数据
            feature_names: 特征名称列表
            
        Returns:
            包含各种评估指标的字典
        """
        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = None
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]  # 获取正类概率
        except:
            pass
        
        # 计算基础指标
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
        }
        
        # 如果有概率预测，计算AUC
        if y_pred_proba is not None:
            try:
                metrics['auc'] = roc_auc_score(y_test, y_pred_proba)
            except:
                metrics['auc'] = 0.0
        
        # 计算混淆矩阵
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
        
        # 交叉验证分数
        try:
            cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='f1')
            metrics['cv_f1_mean'] = float(cv_scores.mean())
            metrics['cv_f1_std'] = float(cv_scores.std())
        except:
            metrics['cv_f1_mean'] = metrics['f1']
            metrics['cv_f1_std'] = 0.0
        
        # 特征重要性
        if feature_names:
            feature_importance = self._get_feature_importance(model, X_test, y_test, feature_names)
            metrics['feature_importance'] = feature_importance
        
        return metrics
    
    def _get_feature_importance(
        self, 
        model, 
        X: np.ndarray, 
        y: np.ndarray, 
        feature_names: List[str]
    ) -> Dict[str, float]:
        """获取特征重要性"""
        try:
            # 尝试直接获取特征重要性
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                return dict(zip(feature_names, importances))
            elif hasattr(model, 'coef_'):
                # 对于线性模型，使用系数的绝对值
                importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
                return dict(zip(feature_names, importances))
            else:
                # 估算特征重要性
                return self._estimate_feature_importance(model, X, y, feature_names)
        except Exception as e:
            logger.warning(f"无法获取特征重要性: {e}")
            # 返回均匀分布的重要性
            return {name: 1.0/len(feature_names) for name in feature_names}
    
    def _estimate_feature_importance(
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
                importances = {k: abs(v)/total_importance for k, v in importances.items()}
            
            return importances
        except Exception as e:
            logger.warning(f"估算特征重要性失败: {e}")
            # 返回均匀分布的重要性
            return {name: 1.0/len(feature_names) for name in feature_names}
    
    def compare_models(self, models_metrics: Dict[str, Dict]) -> Dict[str, any]:
        """
        比较多个模型的性能
        
        Args:
            models_metrics: 不同模型的评估指标字典
            
        Returns:
            包含最佳模型和比较结果的字典
        """
        if not models_metrics:
            return {}
        
        # 找到最佳模型（基于F1分数）
        best_model = max(models_metrics.keys(), 
                        key=lambda k: models_metrics[k].get('f1', 0))
        
        return {
            'best_model': best_model,
            'best_f1': models_metrics[best_model].get('f1', 0),
            'models_ranking': sorted(
                models_metrics.items(), 
                key=lambda item: item[1].get('f1', 0), 
                reverse=True
            )
        }