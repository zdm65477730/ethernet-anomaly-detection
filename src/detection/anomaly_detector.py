import os
import time
import threading
import logging
from queue import Queue, Empty
import numpy as np
from src.system.base_component import BaseComponent
from src.utils.logger import get_logger
from src.models.model_factory import ModelFactory
from src.models.model_selector import ModelSelector
from src.config.config_manager import ConfigManager
from src.features.protocol_specs import get_protocol_spec

class AnomalyDetector(BaseComponent):
    """异常检测器，负责使用模型或规则检测网络流量异常"""
    
    def __init__(self, stat_extractor=None, temporal_extractor=None, config=None):
        super().__init__()
        self.logger = get_logger("anomaly_detector")
        self.config = config or ConfigManager()
        
        # 特征提取器
        self.stat_extractor = stat_extractor
        self.temporal_extractor = temporal_extractor
        
        # 模型相关组件
        self.model_factory = ModelFactory()
        self.model_selector = ModelSelector()
        self.models = {}  # 按协议缓存模型: {protocol_num: model}
        
        # 检测阈值
        self.default_threshold = self.config.get("model.threshold", 0.7)
        self.protocol_thresholds = self.config.get("model.protocol_thresholds", {})
        
        # 特征队列（从traffic_analyzer接收）
        self.features_queue = Queue(maxsize=1000)
        
        # 检测结果队列
        self.results_queue = Queue(maxsize=1000)
        
        # 检测线程
        self._detection_thread = None
        
        # 规则检测配置
        self.detection_rules = self.config.get("detection.rules", {
            "large_packet": 1500,  # 大数据包阈值
            "high_retransmission": 5,  # 重传次数阈值
            "icmp_flood_threshold": 10,  # ICMP包频率阈值
            "tcp_flags": ["SYN+FIN", "SYN+RST", "FIN+URG+PSH"]  # 异常TCP标志
        })
        
        # 协议特定计数器（用于规则检测）
        self.protocol_counters = {}
        
    def set_features_queue(self, queue):
        """设置特征队列（从traffic_analyzer接收）"""
        self.features_queue = queue
        
    def get_results_queue(self):
        """获取检测结果队列"""
        return self.results_queue
    
    def _get_protocol_model(self, protocol_num):
        """获取指定协议的模型，如果没有则加载"""
        if protocol_num in self.models:
            return self.models[protocol_num]
            
        # 选择最佳模型
        model_type = self.model_selector.select_best_model(protocol_num)
        
        # 加载模型
        try:
            model = self.model_factory.load_latest_model(model_type)
            if model:
                self.models[protocol_num] = model
                self.logger.info(f"为协议 {protocol_num} 加载 {model_type} 模型")
                return model
            else:
                self.logger.warning(f"未找到协议 {protocol_num} 的 {model_type} 模型")
                return None
        except Exception as e:
            self.logger.error(f"加载协议 {protocol_num} 的模型时出错: {str(e)}")
            return None
    
    def _get_detection_threshold(self, protocol_num):
        """获取指定协议的检测阈值"""
        return self.protocol_thresholds.get(str(protocol_num), self.default_threshold)
    
    def _rule_based_detection(self, features):
        """基于规则的异常检测（当没有模型时使用）"""
        anomaly_score = 0.0
        anomaly_type = "normal"
        
        protocol_num = features.get("protocol_num", 0)
        protocol_spec = get_protocol_spec(protocol_num)
        protocol_name = protocol_spec["name"]
        
        # 大数据包检测
        if features.get("packet_size", 0) > self.detection_rules.get("large_packet", 1500):
            anomaly_score += 0.4
            anomaly_type = "large_payload"
        
        # TCP异常标志检测
        if protocol_num == 6:  # TCP
            tcp_flags = features.get("tcp_flags", "")
            if tcp_flags in self.detection_rules.get("tcp_flags", []):
                anomaly_score += 0.5
                anomaly_type = "unusual_flags"
                
            # 重传次数过多检测
            if features.get("retransmissions", 0) > self.detection_rules.get("high_retransmission", 5):
                anomaly_score += 0.3
                if anomaly_type == "normal":
                    anomaly_type = "high_retransmissions"
        
        # ICMP Flood检测
        if protocol_num == 1:  # ICMP
            # 更新计数器
            current_time = time.time()
            counter_key = f"icmp_{features.get('src_ip', 'unknown')}"
            
            if counter_key not in self.protocol_counters:
                self.protocol_counters[counter_key] = {"count": 0, "last_reset": current_time}
            
            # 每分钟重置一次计数器
            if current_time - self.protocol_counters[counter_key]["last_reset"] > 60:
                self.protocol_counters[counter_key] = {"count": 1, "last_reset": current_time}
            else:
                self.protocol_counters[counter_key]["count"] += 1
            
            # 检查是否超过阈值
            if self.protocol_counters[counter_key]["count"] > self.detection_rules.get("icmp_flood_threshold", 10):
                anomaly_score += 0.6
                anomaly_type = "icmp_flood"
        
        # 确保分数在0-1之间
        anomaly_score = min(1.0, anomaly_score)
        
        # 如果分数超过阈值，标记为异常
        if anomaly_score >= self._get_detection_threshold(protocol_num):
            return {
                "anomaly_score": anomaly_score,
                "is_anomaly": True,
                "anomaly_type": anomaly_type,
                "detection_method": "rule_based"
            }
        else:
            return {
                "anomaly_score": anomaly_score,
                "is_anomaly": False,
                "anomaly_type": "normal",
                "detection_method": "rule_based"
            }
    
    def _model_based_detection(self, features, protocol_num):
        """基于模型的异常检测"""
        # 提取特征
        feature_names = [k for k, v in features.items() if k not in ["session_id", "protocol", "timestamp"]]
        feature_values = [features[k] for k in feature_names]
        
        # 转换为模型输入格式
        try:
            input_data = np.array(feature_values).reshape(1, -1)
        except Exception as e:
            self.logger.error(f"特征转换失败: {str(e)}")
            return None
        
        # 获取模型
        model = self._get_protocol_model(protocol_num)
        if not model:
            # 模型不存在，使用规则检测
            return self._rule_based_detection(features)
        
        # 模型预测
        try:
            # 对于分类模型，获取异常概率
            if hasattr(model, "predict_proba"):
                # 二分类: [正常概率, 异常概率]
                probabilities = model.predict_proba(input_data)[0]
                anomaly_score = probabilities[1] if len(probabilities) > 1 else probabilities[0]
            else:
                # 对于无概率输出的模型，使用预测结果
                prediction = model.predict(input_data)[0]
                anomaly_score = float(prediction)
            
            # 获取阈值
            threshold = self._get_detection_threshold(protocol_num)
            
            # 确定异常类型（如果模型支持）
            anomaly_type = "unknown_anomaly"
            
            return {
                "anomaly_score": anomaly_score,
                "is_anomaly": anomaly_score >= threshold,
                "anomaly_type": anomaly_type if anomaly_score >= threshold else "normal",
                "detection_method": f"model_{model.model_type}",
                "threshold_used": threshold
            }
            
        except Exception as e:
            self.logger.error(f"模型预测失败: {str(e)}")
            # 预测失败时使用规则检测作为备选
            return self._rule_based_detection(features)
    
    def detect_anomaly(self, features):
        """检测单个特征集是否异常"""
        if not features:
            return None
            
        try:
            # 提取协议信息
            protocol_name = features.get("protocol", "unknown")
            protocol_num = None
            
            # 从协议名称映射到协议号
            protocol_specs = get_protocol_spec.__globals__.get("PROTOCOL_SPECS", {})
            for num, spec in protocol_specs.items():
                if spec["name"] == protocol_name:
                    protocol_num = num
                    break
            
            # 如果找不到协议号，尝试直接使用数字
            if protocol_num is None:
                try:
                    protocol_num = int(protocol_name)
                except:
                    protocol_num = 0  # 未知协议
            
            # 优先使用模型检测
            detection_result = self._model_based_detection(features, protocol_num)
            
            if not detection_result:
                # 模型检测失败，使用规则检测
                detection_result = self._rule_based_detection(features)
            
            # 整合结果
            result = {
                **features, **detection_result,
                "detection_time": time.time()
            }
            
            self.logger.debug(
                f"会话 {features.get('session_id')} 检测结果: "
                f"异常分数={result['anomaly_score']:.4f}, "
                f"是否异常={result['is_anomaly']}, "
                f"检测方法={result['detection_method']}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"异常检测失败: {str(e)}", exc_info=True)
            return None
    
    def _detection_loop(self):
        """检测循环，处理特征队列中的特征并进行异常检测"""
        while self._is_running:
            try:
                # 从队列获取特征
                features = self.features_queue.get(timeout=1)
                if not features:
                    continue
                
                # 执行异常检测
                result = self.detect_anomaly(features)
                if result:
                    # 将结果放入结果队列
                    if not self.results_queue.full():
                        self.results_queue.put(result)
                        
                        # 如果是异常，记录到告警管理器（延迟导入避免循环依赖）
                        if result["is_anomaly"]:
                            from .alert_manager import AlertManager
                            alert_manager = AlertManager(self.config)
                            alert_manager.log_anomaly(
                                anomaly={
                                    "session_id": result.get("session_id"),
                                    "timestamp": result.get("timestamp"),
                                    "anomaly_score": result.get("anomaly_score"),
                                    "anomaly_type": result.get("anomaly_type"),
                                    "src_ip": result.get("src_ip"),
                                    "dst_ip": result.get("dst_ip"),
                                    "protocol": result.get("protocol"),
                                    "protocol_name": result.get("protocol")
                                },
                                features=features
                            )
                
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
            return
            
        super().start()
        
        # 启动检测线程
        self._detection_thread = threading.Thread(
            target=self._detection_loop,
            daemon=True
        )
        self._detection_thread.start()
        
        self.logger.info("异常检测器已启动")
    
    def stop(self):
        """停止异常检测器"""
        if not self._is_running:
            return
            
        super().stop()
        
        # 停止检测线程
        if self._detection_thread and self._detection_thread.is_alive():
            self._detection_thread.join(timeout=5)
            if self._detection_thread.is_alive():
                self.logger.warning("检测线程未能正常终止")
        
        self.logger.info("异常检测器已停止")
    
    def get_status(self):
        """获取组件状态"""
        status = super().get_status()
        status.update({
            "features_queue_size": self.features_queue.qsize(),
            "results_queue_size": self.results_queue.qsize(),
            "loaded_models": list(self.models.keys()),
            "default_threshold": self.default_threshold,
            "protocol_thresholds": self.protocol_thresholds
        })
        return status