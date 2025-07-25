import os
import time
import json
import threading
from datetime import datetime
from queue import Queue, Empty
import pandas as pd
import numpy as np
from src.system.base_component import BaseComponent
from src.utils.logger import get_logger
from src.config.config_manager import ConfigManager

class FeedbackProcessor(BaseComponent):
    """反馈处理器，处理人工对异常检测结果的反馈，用于优化模型和特征"""
    
    def __init__(self, feature_extractor=None, config=None):
        super().__init__()
        self.logger = get_logger("feedback_processor")
        self.config = config or ConfigManager()
        
        # 特征提取器（用于特征重要性分析）
        self.feature_extractor = feature_extractor
        
        # 反馈队列
        self.feedback_queue = Queue(maxsize=1000)
        
        # 反馈存储目录
        self.feedback_dir = self.config.get("feedback.dir", "data/feedback")
        os.makedirs(self.feedback_dir, exist_ok=True)
        
        # 特征性能统计
        self.feature_performance = {}  # {feature_name: {tp: int, fp: int, tn: int, fn: int}}
        
        # 协议性能统计
        self.protocol_performance = {}  # {protocol: {tp: int, fp: int, tn: int, fn: int}}
        
        # 处理线程
        self._processing_thread = None
        
        # 加载已有反馈
        self._load_existing_feedback()
        
    def _load_existing_feedback(self):
        """加载已有的反馈数据"""
        try:
            # 遍历反馈目录
            for filename in os.listdir(self.feedback_dir):
                if not filename.endswith(".json"):
                    continue
                    
                file_path = os.path.join(self.feedback_dir, filename)
                with open(file_path, "r") as f:
                    feedbacks = json.load(f)
                    
                # 处理每个反馈
                for feedback in feedbacks:
                    self._process_feedback_item(feedback, save_to_disk=False)
                    
            self.logger.info(f"已加载 {len(os.listdir(self.feedback_dir))} 个反馈文件")
        except Exception as e:
            self.logger.error(f"加载已有反馈时出错: {str(e)}")
    
    def submit_feedback(self, feedback):
        """
        提交反馈
        
        参数:
            feedback: 反馈字典，包含:
                - anomaly_id: 异常ID
                - session_id: 会话ID
                - is_correct: 检测结果是否正确 (True/False)
                - actual_type: 实际异常类型 (如果is_correct为False)
                - features: 检测时使用的特征
                - detection_result: 原始检测结果
                - timestamp: 反馈提交时间
        """
        if not feedback or "anomaly_id" not in feedback or "is_correct" not in feedback:
            self.logger.warning("无效的反馈数据")
            return False
            
        # 添加提交时间
        if "timestamp" not in feedback:
            feedback["timestamp"] = time.time()
            
        # 将反馈放入队列
        try:
            self.feedback_queue.put(feedback, block=False)
            self.logger.debug(f"已接收反馈: {feedback['anomaly_id']}")
            return True
        except Exception as e:
            self.logger.error(f"提交反馈失败: {str(e)}")
            return False
    
    def _process_feedback_item(self, feedback, save_to_disk=True):
        """处理单个反馈项"""
        try:
            anomaly_id = feedback["anomaly_id"]
            is_correct = feedback["is_correct"]
            features = feedback.get("features", {})
            detection_result = feedback.get("detection_result", {})
            protocol = detection_result.get("protocol", "unknown")
            
            # 提取检测结果
            predicted_anomaly = detection_result.get("is_anomaly", False)
            anomaly_score = detection_result.get("anomaly_score", 0)
            
            # 确定真实标签（基于反馈）
            actual_anomaly = not is_correct if not predicted_anomaly else is_correct
            
            # 更新协议性能统计
            if protocol not in self.protocol_performance:
                self.protocol_performance[protocol] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
            
            # 真阳性 (TP): 预测为异常，实际为异常
            if predicted_anomaly and actual_anomaly:
                self.protocol_performance[protocol]["tp"] += 1
            # 假阳性 (FP): 预测为异常，实际为正常
            elif predicted_anomaly and not actual_anomaly:
                self.protocol_performance[protocol]["fp"] += 1
            # 真阴性 (TN): 预测为正常，实际为正常
            elif not predicted_anomaly and not actual_anomaly:
                self.protocol_performance[protocol]["tn"] += 1
            # 假阴性 (FN): 预测为正常，实际为异常
            elif not predicted_anomaly and actual_anomaly:
                self.protocol_performance[protocol]["fn"] += 1
            
            # 更新特征性能统计
            for feature_name, feature_value in features.items():
                if feature_name in ["session_id", "protocol", "timestamp", "src_ip", "dst_ip"]:
                    continue
                    
                if feature_name not in self.feature_performance:
                    self.feature_performance[feature_name] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
                
                # 根据特征值和检测结果更新统计
                # 这里简化处理，实际应用中可能需要更复杂的逻辑
                self.feature_performance[feature_name][
                    "tp" if (predicted_anomaly and actual_anomaly) else
                    "fp" if (predicted_anomaly and not actual_anomaly) else
                    "tn" if (not predicted_anomaly and not actual_anomaly) else
                    "fn"
                ] += 1
            
            self.logger.debug(f"已处理反馈: {anomaly_id}, 正确: {is_correct}")
            
            # 保存反馈到磁盘
            if save_to_disk:
                date_str = datetime.fromtimestamp(feedback["timestamp"]).strftime("%Y%m%d")
                feedback_file = os.path.join(self.feedback_dir, f"feedback_{date_str}.json")
                
                # 读取已有反馈
                existing_feedbacks = []
                if os.path.exists(feedback_file):
                    with open(feedback_file, "r") as f:
                        existing_feedbacks = json.load(f)
                
                # 添加新反馈
                existing_feedbacks.append(feedback)
                
                # 保存更新后的反馈
                with open(feedback_file, "w") as f:
                    json.dump(existing_feedbacks, f, indent=2, ensure_ascii=False)
            
            # 如果有特征提取器，通知其更新特征重要性
            if self.feature_extractor and hasattr(self.feature_extractor, "update_feature_performance"):
                self.feature_extractor.update_feature_performance(self.feature_performance)
                
            return True
            
        except Exception as e:
            self.logger.error(f"处理反馈项时出错: {str(e)}", exc_info=True)
            return False
    
    def _processing_loop(self):
        """处理反馈队列中的反馈项"""
        while self._is_running:
            try:
                # 从队列获取反馈
                feedback = self.feedback_queue.get(timeout=1)
                if not feedback:
                    continue
                
                # 处理反馈
                self._process_feedback_item(feedback)
                
                # 标记任务完成
                self.feedback_queue.task_done()
                
            except Empty:
                # 队列为空，继续循环
                continue
            except Exception as e:
                self.logger.error(f"反馈处理循环出错: {str(e)}", exc_info=True)
                time.sleep(1)
    
    def get_feature_metrics(self, feature_name=None):
        """
        获取特征的性能指标
        
        参数:
            feature_name: 特征名称，为None则返回所有特征
            
        返回:
            包含精确率、召回率、F1等指标的字典
        """
        metrics = {}
        
        # 计算单个特征的指标
        def calculate_metrics(performance):
            tp = performance.get("tp", 0)
            fp = performance.get("fp", 0)
            tn = performance.get("tn", 0)
            fn = performance.get("fn", 0)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # 假阳性率
            
            return {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fpr": fpr,
                "total": tp + fp + tn + fn
            }
        
        if feature_name:
            if feature_name in self.feature_performance:
                return {feature_name: calculate_metrics(self.feature_performance[feature_name])}
            else:
                return {}
        else:
            # 计算所有特征的指标
            for feature, performance in self.feature_performance.items():
                metrics[feature] = calculate_metrics(performance)
            
            return metrics
    
    def get_low_performance_features(self, threshold=0.5):
        """获取性能较差的特征（F1 < threshold）"""
        all_metrics = self.get_feature_metrics()
        return {
            feature: metrics for feature, metrics in all_metrics.items()
            if metrics["f1"] < threshold and metrics["total"] > 10  # 至少有10个样本
        }
    
    def get_protocol_metrics(self, protocol=None):
        """获取协议的检测性能指标"""
        metrics = {}
        
        # 计算单个协议的指标
        def calculate_metrics(performance):
            tp = performance.get("tp", 0)
            fp = performance.get("fp", 0)
            tn = performance.get("tn", 0)
            fn = performance.get("fn", 0)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
            
            return {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
                "total": tp + fp + tn + fn
            }
        
        if protocol:
            if protocol in self.protocol_performance:
                return {protocol: calculate_metrics(self.protocol_performance[protocol])}
            else:
                return {}
        else:
            # 计算所有协议的指标
            for proto, performance in self.protocol_performance.items():
                metrics[proto] = calculate_metrics(performance)
            
            return metrics
    
    def start(self):
        """启动反馈处理器"""
        if self._is_running:
            self.logger.warning("反馈处理器已在运行中")
            return
            
        super().start()
        
        # 启动处理线程
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self._processing_thread.start()
        
        self.logger.info("反馈处理器已启动")
    
    def stop(self):
        """停止反馈处理器"""
        if not self._is_running:
            return
            
        super().stop()
        
        # 停止处理线程
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5)
            if self._processing_thread.is_alive():
                self.logger.warning("反馈处理线程未能正常终止")
        
        self.logger.info("反馈处理器已停止")
    
    def get_status(self):
        """获取组件状态"""
        status = super().get_status()
        status.update({
            "feedback_queue_size": self.feedback_queue.qsize(),
            "feedback_dir": self.feedback_dir,
            "feature_count": len(self.feature_performance),
            "protocol_count": len(self.protocol_performance),
            "total_feedback": sum(
                sum(perf.values()) for perf in self.feature_performance.values()
            ) // 4  # 每个反馈会更新4个指标
        })
        return status
    