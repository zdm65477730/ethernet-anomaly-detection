import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, confusion_matrix
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

class EvaluationVisualizer:
    """评估可视化工具，生成混淆矩阵、ROC曲线等评估图表"""
    
    def __init__(self, output_dir: str = "evaluation_plots"):
        """
        初始化可视化工具
        
        参数:
            output_dir: 图表输出目录
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置默认样式
        sns.set_style("whitegrid")
        plt.style.use("seaborn-v0_8-colorblind")
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        title: str = "混淆矩阵",
        normalize: bool = False,
        save_name: Optional[str] = None,
        show: bool = True
    ) -> str:
        """
        绘制混淆矩阵
        
        参数:
            cm: 混淆矩阵数组
            class_names: 类别名称列表
            title: 图表标题
            normalize: 是否归一化
            save_name: 保存文件名，为None则自动生成
            show: 是否显示图表
            
        返回:
            保存文件路径
        """
        plt.figure(figsize=(10, 8))
        
        # 归一化处理
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title += " (归一化)"
        else:
            fmt = 'd'
        
        # 绘制热图
        ax = sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", 
                        xticklabels=class_names, yticklabels=class_names)
        
        # 设置标题和标签
        plt.title(title)
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if not save_name:
            save_name = f"confusion_matrix_{'normalized' if normalize else 'raw'}.png"
        file_path = os.path.join(self.output_dir, save_name)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return file_path
    
    def plot_roc_curve(
        self,
        y_true: Union[np.ndarray, List[int]],
        y_score: Union[np.ndarray, List[float]],
        title: str = "ROC曲线",
        save_name: Optional[str] = None,
        show: bool = True,
        auc_score: Optional[float] = None
    ) -> str:
        """
        绘制ROC曲线
        
        参数:
            y_true: 真实标签
            y_score: 预测概率分数
            title: 图表标题
            save_name: 保存文件名
            show: 是否显示图表
            auc_score: 已计算的AUC值，为None则自动计算
            
        返回:
            保存文件路径
        """
        # 计算ROC曲线数据
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        
        # 计算AUC
        if auc_score is None:
            from .metrics import calculate_auc
            auc_score = calculate_auc(y_true, y_score)
        
        # 绘制ROC曲线
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC曲线 (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        # 设置坐标轴范围和标签
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正例率 (FPR)')
        plt.ylabel('真正例率 (TPR)')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        if not save_name:
            save_name = "roc_curve.png"
        file_path = os.path.join(self.output_dir, save_name)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return file_path
    
    def plot_precision_recall_curve(
        self,
        y_true: Union[np.ndarray, List[int]],
        y_score: Union[np.ndarray, List[float]],
        title: str = "精确率-召回率曲线",
        save_name: Optional[str] = None,
        show: bool = True,
        average_precision: Optional[float] = None
    ) -> str:
        """
        绘制精确率-召回率曲线
        
        参数:
            y_true: 真实标签
            y_score: 预测概率分数
            title: 图表标题
            save_name: 保存文件名
            show: 是否显示图表
            average_precision: 平均精确率，为None则自动计算
            
        返回:
            保存文件路径
        """
        # 计算精确率-召回率曲线数据
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        
        # 计算平均精确率
        if average_precision is None:
            average_precision = average_precision_score(y_true, y_score)
        
        # 绘制曲线
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'精确率-召回率曲线 (AP = {average_precision:.3f})')
        
        # 设置坐标轴范围和标签
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('召回率 (Recall)')
        plt.ylabel('精确率 (Precision)')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        if not save_name:
            save_name = "precision_recall_curve.png"
        file_path = os.path.join(self.output_dir, save_name)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return file_path
    
    def plot_feature_importance(
        self,
        importance: Dict[str, float],
        title: str = "特征重要性",
        top_n: Optional[int] = None,
        save_name: Optional[str] = None,
        show: bool = True
    ) -> str:
        """
        绘制特征重要性条形图
        
        参数:
            importance: 特征重要性字典 {特征名: 重要性值}
            title: 图表标题
            top_n: 只显示前N个重要特征，为None则显示所有
            save_name: 保存文件名
            show: 是否显示图表
            
        返回:
            保存文件路径
        """
        # 排序特征重要性
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        # 取前N个特征
        if top_n and len(sorted_importance) > top_n:
            sorted_importance = sorted_importance[:top_n]
            title += f" (前{top_n}个)"
        
        # 提取特征名和重要性值
        features, values = zip(*sorted_importance)
        
        # 绘制条形图
        plt.figure(figsize=(12, 8))
        y_pos = np.arange(len(features))
        plt.barh(y_pos, values, align='center', color='skyblue')
        plt.yticks(y_pos, features)
        plt.xlabel('重要性值')
        plt.title(title)
        plt.gca().invert_yaxis()  # 最重要的特征在顶部
        plt.grid(True, axis='x', alpha=0.3)
        
        # 保存图表
        if not save_name:
            save_name = "feature_importance.png"
        file_path = os.path.join(self.output_dir, save_name)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return file_path
    
    def plot_metrics_comparison(
        self,
        metrics: Dict[str, Dict[str, float]],
        title: str = "模型指标对比",
        save_name: Optional[str] = None,
        show: bool = True
    ) -> str:
        """
        绘制多个模型的指标对比图
        
        参数:
            metrics: 模型指标字典 {模型名: {指标名: 指标值}}
            title: 图表标题
            save_name: 保存文件名
            show: 是否显示图表
            
        返回:
            保存文件路径
        """
        # 转换数据格式
        df = pd.DataFrame(metrics).T
        
        # 绘制条形图
        plt.figure(figsize=(12, 8))
        ax = df.plot(kind='bar', width=0.8)
        
        # 设置标签和标题
        plt.title(title)
        plt.ylabel('指标值')
        plt.xlabel('模型')
        plt.ylim(0, 1.05)
        plt.grid(True, axis='y', alpha=0.3)
        
        # 添加数值标签
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.3f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=9, color='black', 
                        xytext=(0, 5), textcoords='offset points')
        
        # 调整布局
        plt.legend(title='指标', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # 保存图表
        if not save_name:
            save_name = "metrics_comparison.png"
        file_path = os.path.join(self.output_dir, save_name)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return file_path
    
    def plot_protocol_metrics(
        self,
        protocol_metrics: Dict[str, Dict[str, float]],
        title: str = "各协议性能指标",
        save_name: Optional[str] = None,
        show: bool = True
    ) -> str:
        """
        绘制各协议的性能指标对比图
        
        参数:
            protocol_metrics: 协议指标字典 {协议名: {指标名: 指标值}}
            title: 图表标题
            save_name: 保存文件名
            show: 是否显示图表
            
        返回:
            保存文件路径
        """
        # 转换数据格式
        df = pd.DataFrame(protocol_metrics).T
        
        # 绘制条形图
        plt.figure(figsize=(14, 8))
        ax = df[['precision', 'recall', 'f1']].plot(kind='bar', width=0.8)
        
        # 设置标签和标题
        plt.title(title)
        plt.ylabel('指标值')
        plt.xlabel('协议')
        plt.ylim(0, 1.05)
        plt.grid(True, axis='y', alpha=0.3)
        
        # 添加样本数标签
        if 'support' in df.columns:
            for i, support in enumerate(df['support']):
                plt.text(i, -0.05, f"n={support}", ha='center', va='top', fontsize=9)
        
        # 调整布局
        plt.legend(title='指标', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # 保存图表
        if not save_name:
            save_name = "protocol_metrics.png"
        file_path = os.path.join(self.output_dir, save_name)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return file_path
    
    def generate_evaluation_report(
        self,
        y_true: Union[np.ndarray, List[int]],
        y_pred: Union[np.ndarray, List[int]],
        y_score: Union[np.ndarray, List[float]],
        class_names: List[str] = ["正常", "异常"],
        feature_importance: Optional[Dict[str, float]] = None,
        protocol_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        report_dir: Optional[str] = None,
        model_name: str = "模型"
    ) -> Dict[str, str]:
        """
        生成完整的评估报告，包含多种图表
        
        参数:
            y_true: 真实标签
            y_pred: 预测标签
            y_score: 预测概率分数
            class_names: 类别名称
            feature_importance: 特征重要性字典
            protocol_metrics: 协议指标字典
            report_dir: 报告目录，为None则使用默认目录
            model_name: 模型名称
            
        返回:
            包含所有生成图表路径的字典
        """
        # 创建报告目录
        report_dir = report_dir or os.path.join(self.output_dir, f"{model_name}_evaluation")
        os.makedirs(report_dir, exist_ok=True)
        self.output_dir = report_dir
        
        # 计算混淆矩阵
        from .metrics import calculate_confusion_matrix
        cm, _ = calculate_confusion_matrix(y_true, y_pred)
        
        # 计算AUC
        from .metrics import calculate_auc
        auc_score = calculate_auc(y_true, y_score)
        
        # 生成各种图表
        plots = {}
        
        plots["confusion_matrix"] = self.plot_confusion_matrix(
            cm, class_names, 
            title=f"{model_name} 混淆矩阵",
            save_name="confusion_matrix.png",
            show=False
        )
        
        plots["confusion_matrix_normalized"] = self.plot_confusion_matrix(
            cm, class_names, 
            title=f"{model_name} 混淆矩阵",
            normalize=True,
            save_name="confusion_matrix_normalized.png",
            show=False
        )
        
        plots["roc_curve"] = self.plot_roc_curve(
            y_true, y_score,
            title=f"{model_name} ROC曲线",
            save_name="roc_curve.png",
            show=False,
            auc_score=auc_score
        )
        
        plots["precision_recall_curve"] = self.plot_precision_recall_curve(
            y_true, y_score,
            title=f"{model_name} 精确率-召回率曲线",
            save_name="precision_recall_curve.png",
            show=False
        )
        
        # 如果提供了特征重要性，生成特征重要性图
        if feature_importance:
            plots["feature_importance"] = self.plot_feature_importance(
                feature_importance,
                title=f"{model_name} 特征重要性",
                save_name="feature_importance.png",
                show=False
            )
        
        # 如果提供了协议指标，生成协议指标图
        if protocol_metrics:
            plots["protocol_metrics"] = self.plot_protocol_metrics(
                protocol_metrics,
                title=f"{model_name} 各协议性能指标",
                save_name="protocol_metrics.png",
                show=False
            )
        
        return plots
