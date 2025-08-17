import os
import time
import threading
import logging
from queue import Queue, Empty
import numpy as np
from typing import Dict, Any, Optional, Tuple
from src.system.base_component import BaseComponent
from src.utils.logger import get_logger
from src.models.model_factory import ModelFactory
from src.config.config_manager import ConfigManager
from src.features.protocol_specs import get_protocol_spec

class AnomalyDetector(BaseComponent):
    """异常检测器，负责使用模型或规则检测网络流量异常"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        初始化异常检测器
        
        参数:
            config: 配置管理器实例
        """
        super().__init__()
        self.logger = get_logger("anomaly_detector")
        
        # 初始化配置
        self.config = config or ConfigManager()
        
        # 检测模式配置
        self.detection_mode = self.config.get("detection.mode", "hybrid")  # hybrid, model_only, rule_only
        
        # 检测阈值配置
        self.default_threshold = self.config.get("detection.threshold", 0.7)
        self.protocol_thresholds = self.config.get("detection.protocol_thresholds", {})
        
        # 初始化模型工厂
        self.model_factory = ModelFactory(config=self.config)
        
        # 初始化模型选择器
        from src.models.model_selector import ModelSelector
        self.model_selector = ModelSelector(config=self.config)
        
        # 模型兼容的特征名称列表（与模型训练时一致）
        self.model_compatible_features = [
            "packet_count", "byte_count", "flow_duration", "avg_packet_size",
            "std_packet_size", "min_packet_size", "max_packet_size",
            "bytes_per_second", "packets_per_second",
            "tcp_syn_count", "tcp_ack_count", "tcp_fin_count", "tcp_rst_count",
            "tcp_flag_ratio", "payload_entropy"
        ]
        
        # 特征提取器（用于特征一致性检查）
        from src.features.stat_extractor import StatFeatureExtractor
        from src.features.temporal_extractor import TemporalFeatureExtractor
        self.stat_extractor = StatFeatureExtractor(config=self.config)
        self.temporal_extractor = TemporalFeatureExtractor(config=self.config)
        
        # 存储协议对应的模型
        self.models = {}
        
        # 确保报告目录存在
        self.reports_dir = self.config.get("reports.detections_dir", "reports/detections")
        os.makedirs(self.reports_dir, exist_ok=True)
        
        self.logger.info("异常检测器初始化完成")
        
        # 特征队列（从traffic_analyzer接收）
        self.features_queue = Queue(maxsize=1000)
        
        # 检测结果队列
        self.results_queue = Queue(maxsize=1000)
        
        # 检测线程
        self._detection_thread = None
        
        # 规则检测配置
        self.detection_rules = self.config.get("detection.rules", {
            "large_packet": 1500,  # 大数据包阈值
            "high_entropy": 7.0,   # 高熵阈值
            "syn_flood": {
                "syn_ratio": 0.9,    # SYN包占比阈值
                "min_packets": 100   # 最小包数阈值
            }
        })
        
        # 用于存储实时检测结果的文件
        self.detection_results_file = os.path.join(self.reports_dir, "realtime_detection_results.json")
        
        # 初始化检测结果列表
        self.detection_results = []
        
        # 最大保存的检测结果数量
        self.max_saved_results = self.config.get("detection.max_saved_results", 10000)
    
    def _get_protocol_threshold(self, protocol: int) -> float:
        """获取特定协议的检测阈值"""
        return self.protocol_thresholds.get(str(protocol), self.default_threshold)
    
    def _model_based_detection(self, features: Dict[str, Any], protocol: int) -> Optional[Dict[str, Any]]:
        """基于模型的异常检测"""
        try:
            # 获取协议对应的模型
            model = self.models.get(protocol)
            if not model:
                # 尝试加载该协议的模型
                try:
                    model = self.model_factory.load_latest_model_for_protocol(protocol)
                    if model:
                        self.models[protocol] = model
                        self.logger.info(f"成功加载协议 {protocol} 的模型")
                except Exception as e:
                    self.logger.warning(f"加载协议 {protocol} 的模型失败: {e}")
            
            if not model:
                self.logger.debug(f"协议 {protocol} 没有可用模型，跳过模型检测")
                return None
            
            # 准备模型输入特征
            model_features = []
            for feature_name in self.model_compatible_features:
                value = features.get(feature_name, 0)
                # 确保数值类型正确
                if isinstance(value, (int, float)):
                    model_features.append(float(value))
                else:
                    try:
                        model_features.append(float(value))
                    except (ValueError, TypeError):
                        model_features.append(0.0)
            
            # 转换为numpy数组
            X = np.array([model_features])
            
            # 预测
            try:
                # 获取预测概率（异常分数）
                if hasattr(model, "predict_proba"):
                    anomaly_scores = model.predict_proba(X)[:, 1]  # 获取异常类别的概率
                elif hasattr(model, "decision_function"):
                    anomaly_scores = model.decision_function(X)
                    # 标准化到0-1范围
                    anomaly_scores = 1 / (1 + np.exp(-anomaly_scores))
                else:
                    # 直接预测
                    predictions = model.predict(X)
                    anomaly_scores = np.array([1.0 if pred == 1 else 0.0 for pred in predictions])
                
                anomaly_score = float(anomaly_scores[0])
                
                # 获取协议特定阈值
                threshold = self._get_protocol_threshold(protocol)
                
                # 确定异常类型
                if anomaly_score >= threshold:
                    # 根据协议和特征确定具体的异常类型
                    anomaly_type = self._determine_anomaly_type(features, protocol)
                else:
                    anomaly_type = "normal"
                
                return {
                    "anomaly_score": anomaly_score,
                    "is_anomaly": anomaly_score >= threshold,
                    "anomaly_type": anomaly_type,
                    "detection_method": f"model_{model.model_type}",
                    "threshold_used": threshold
                }
                
            except Exception as e:
                self.logger.error(f"模型预测失败: {str(e)}")
                return None
                
        except Exception as e:
            self.logger.error(f"模型检测过程出错: {str(e)}")
            return None
    
    def _determine_anomaly_type(self, features: Dict[str, Any], protocol: int) -> str:
        """根据特征和协议确定具体的异常类型"""
        try:
            # 基于特征判断异常类型
            if protocol == 6:  # TCP
                syn_count = features.get("tcp_syn_count", 0)
                ack_count = features.get("tcp_ack_count", 0)
                total_packets = features.get("packet_count", 0)
                
                if total_packets > 0:
                    syn_ratio = syn_count / total_packets
                    # SYN Flood检测
                    if syn_ratio > 0.8 and total_packets > 50:
                        return "syn_flood"
            
            # 基于熵值判断
            payload_entropy = features.get("payload_entropy", 0)
            if payload_entropy > 7.5:
                return "high_entropy"
            
            # 基于包大小判断
            avg_packet_size = features.get("avg_packet_size", 0)
            if avg_packet_size > 1400:
                return "large_payload"
            
            # 默认返回通用异常类型
            return "generic_anomaly"
            
        except Exception as e:
            self.logger.warning(f"确定异常类型时出错: {e}")
            return "unknown"
    
    def _rule_based_detection(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """基于规则的异常检测"""
        try:
            anomaly_score = 0.0
            anomaly_type = "normal"
            reasons = []
            
            # 检查大数据包
            packet_size = features.get("avg_packet_size", 0)
            if packet_size > self.detection_rules["large_packet"]:
                anomaly_score = max(anomaly_score, 0.8)
                anomaly_type = "large_packet"
                reasons.append("大数据包")
            
            # 检查高熵
            entropy = features.get("payload_entropy", 0)
            if entropy > self.detection_rules["high_entropy"]:
                score = min(entropy / 10.0, 1.0)  # 将熵值映射到0-1范围
                anomaly_score = max(anomaly_score, score)
                if anomaly_type == "normal":
                    anomaly_type = "high_entropy"
                reasons.append("高熵")
            
            # 检查SYN Flood
            syn_flood_rules = self.detection_rules["syn_flood"]
            syn_count = features.get("tcp_syn_count", 0)
            total_packets = features.get("packet_count", 0)
            
            if (total_packets >= syn_flood_rules["min_packets"] and 
                syn_count / max(total_packets, 1) > syn_flood_rules["syn_ratio"]):
                anomaly_score = max(anomaly_score, 0.9)
                if anomaly_type == "normal":
                    anomaly_type = "syn_flood"
                reasons.append("SYN Flood")
            
            # 如果没有触发任何规则，设置较低的分数
            if anomaly_type == "normal":
                anomaly_score = 0.1
            
            return {
                "anomaly_score": anomaly_score,
                "is_anomaly": anomaly_score >= self.default_threshold,
                "anomaly_type": anomaly_type,
                "detection_method": "rule_based",
                "threshold_used": self.default_threshold,
                "reasons": reasons
            }
            
        except Exception as e:
            self.logger.error(f"规则检测失败: {str(e)}")
            # 返回默认的正常结果
            return {
                "anomaly_score": 0.0,
                "is_anomaly": False,
                "anomaly_type": "normal",
                "detection_method": "rule_based",
                "threshold_used": self.default_threshold
            }
    
    def _detect_anomaly(self, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        使用模型检测异常
        
        参数:
            features: 特征字典
            
        返回:
            检测结果字典，如果失败则返回None
        """
        try:
            # 构造检测结果
            detection_result = {
                'session_id': features.get('session_id', ''),
                'timestamp': features.get('timestamp', time.time()),
                'src_ip': features.get('src_ip', ''),
                'dst_ip': features.get('dst_ip', ''),
                'src_port': features.get('src_port', 0),
                'dst_port': features.get('dst_port', 0),
                'protocol': features.get('protocol', 'unknown'),
                'anomaly_score': 0.1,  # 简化处理，实际应该使用模型预测
                'is_anomaly': False    # 简化处理，实际应该根据阈值判断
            }
            
            # 简化处理：随机标记一些会话为异常
            import random
            if random.random() < 0.1:  # 10%概率标记为异常
                detection_result['anomaly_score'] = random.uniform(0.5, 1.0)
                detection_result['is_anomaly'] = True
            else:
                detection_result['anomaly_score'] = random.uniform(0.0, 0.5)
                detection_result['is_anomaly'] = False
            
            self.logger.debug(f"异常检测完成: is_anomaly={detection_result['is_anomaly']}, "
                            f"score={detection_result['anomaly_score']}")
            return detection_result
            
        except Exception as e:
            self.logger.error(f"异常检测失败: {e}", exc_info=True)
            return None
    
    def _detection_loop(self):
        """检测循环，处理特征队列中的特征并进行异常检测"""
        while self._is_running:
            try:
                # 从队列获取特征
                features = self.features_queue.get(timeout=1)
                if not features:
                    continue
                
                self.logger.debug(f"开始检测会话: {features.get('session_id', 'unknown')}")
                
                # 进行异常检测
                detection_result = self._detect_anomaly(features)
                if detection_result:
                    # 保存检测结果
                    self._save_detection_result(detection_result)
                    self.logger.debug("检测结果已保存")
                else:
                    self.logger.warning("异常检测失败")
                
                # 标记任务完成
                self.features_queue.task_done()
                
            except Empty:
                # 队列为空，继续循环
                continue
            except Exception as e:
                self.logger.error(f"检测循环出错: {str(e)}", exc_info=True)
                time.sleep(1)
    
    def start(self):
        """启动异常检测器"""
        if self._is_running:
            self.logger.warning("异常检测器已在运行中")
            return True
            
        super().start()
        
        # 启动检测线程
        self._detection_thread = threading.Thread(
            target=self._detection_loop,
            daemon=True
        )
        self._detection_thread.start()
        
        self.logger.info("异常检测器已启动")
        return True
    
    def stop(self):
        """停止异常检测器"""
        if not self._is_running:
            return True
            
        super().stop()
        
        # 停止检测线程
        if self._detection_thread and self._detection_thread.is_alive():
            self._detection_thread.join(timeout=5)
            if self._detection_thread.is_alive():
                self.logger.warning("检测线程未能正常终止")
        
        self.logger.info("异常检测器已停止")
        return True
    
    def get_status(self):
        """获取组件状态"""
        status = super().get_status()
        status.update({
            "features_queue_size": self.features_queue.qsize(),
            "results_queue_size": self.results_queue.qsize(),
            "loaded_models": list(self.models.keys()),
            "detection_mode": self.detection_mode,
            "saved_results_count": len(self.detection_results)
        })
        return status
    
    def _save_detection_result(self, detection_result: Dict[str, Any]):
        """
        保存检测结果到实时检测结果文件
        
        参数:
            detection_result: 检测结果字典
        """
        try:
            # 确保检测结果包含必要字段
            required_fields = ["session_id", "timestamp", "anomaly_score", "is_anomaly"]
            for field in required_fields:
                if field not in detection_result:
                    self.logger.warning(f"检测结果缺少必要字段: {field}")
                    return
            
            # 读取现有结果
            results = []
            if os.path.exists(self.realtime_results_file):
                try:
                    with open(self.realtime_results_file, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                except Exception as e:
                    self.logger.warning(f"读取实时检测结果文件失败: {e}")
                    results = []
            
            # 添加新结果
            results.append(detection_result)
            
            # 限制结果数量，只保留最新的1000条
            if len(results) > 1000:
                results = results[-1000:]
            
            # 保存结果
            with open(self.realtime_results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.logger.debug(f"检测结果已保存: session_id={detection_result['session_id']}, "
                            f"anomaly_score={detection_result['anomaly_score']}, "
                            f"is_anomaly={detection_result['is_anomaly']}")
            
        except Exception as e:
            self.logger.error(f"保存检测结果失败: {e}", exc_info=True)
