from typing import List, Tuple, Optional, Union, Any
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report as sk_classification_report
)
from typing import Tuple, Dict, List, Union

def calculate_precision(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    average: str = "binary",
    zero_division: int = 0
) -> float:
    """
    计算精确率 (Precision)
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        average: 多类别的平均方式
        zero_division: 当出现零除时的返回值
        
    返回:
        精确率值
    """
    return precision_score(
        y_true, 
        y_pred, 
        average=average, 
        zero_division=zero_division
    )

def calculate_recall(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    average: str = "binary",
    zero_division: int = 0
) -> float:
    """
    计算召回率 (Recall)
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        average: 多类别的平均方式
        zero_division: 当出现零除时的返回值
        
    返回:
        召回率值
    """
    return recall_score(
        y_true, 
        y_pred, 
        average=average, 
        zero_division=zero_division
    )

def calculate_f1_score(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    average: str = "binary",
    zero_division: int = 0
) -> float:
    """
    计算F1分数
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        average: 多类别的平均方式
        zero_division: 当出现零除时的返回值
        
    返回:
        F1分数值
    """
    return f1_score(
        y_true, 
        y_pred, 
        average=average, 
        zero_division=zero_division
    )

def calculate_auc(
    y_true: Union[np.ndarray, List[int]],
    y_score: Union[np.ndarray, List[float]]
) -> float:
    """
    计算AUC值
    
    参数:
        y_true: 真实标签
        y_score: 预测的概率分数
        
    返回:
        AUC值
    """
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        # 处理只有一个类别的情况
        return 0.0

def calculate_confusion_matrix(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]]
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    计算混淆矩阵
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        
    返回:
        混淆矩阵数组和包含TP, TN, FP, FN的字典
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # 处理二分类情况
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics = {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        }
        return cm, metrics
    else:
        # 多分类情况，返回总计数
        total = np.sum(cm)
        correct = np.sum(np.diag(cm))
        incorrect = total - correct
        return cm, {
            "total_samples": int(total),
            "correct_predictions": int(correct),
            "incorrect_predictions": int(incorrect)
        }

def classification_report(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    target_names: Optional[List[str]] = None,
    output_dict: bool = False
) -> Union[str, Dict[str, Any]]:
    """
    生成完整的分类报告
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        target_names: 类别名称列表
        output_dict: 是否返回字典格式
        
    返回:
        分类报告字符串或字典
    """
    return sk_classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=output_dict,
        zero_division=0
    )

def calculate_protocol_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    protocols: Union[np.ndarray, List[int]],
    protocol_names: Optional[Dict[int, str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    按协议计算评估指标
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        protocols: 每个样本对应的协议
        protocol_names: 协议编号到名称的映射
        
    返回:
        按协议分组的评估指标字典
    """
    results = {}
    unique_protocols = np.unique(protocols)
    
    for proto in unique_protocols:
        # 筛选该协议的样本
        mask = protocols == proto
        proto_y_true = y_true[mask]
        proto_y_pred = y_pred[mask]
        
        # 计算指标
        precision = calculate_precision(proto_y_true, proto_y_pred)
        recall = calculate_recall(proto_y_true, proto_y_pred)
        f1 = calculate_f1_score(proto_y_true, proto_y_pred)
        
        # 确定协议名称
        proto_name = str(proto)
        if protocol_names and proto in protocol_names:
            proto_name = protocol_names[proto]
        
        results[proto_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": len(proto_y_true)
        }
    
    return results
