import time
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from src.utils.logger import get_logger
from src.config.config_manager import ConfigManager
from src.data.data_storage import DataStorage
from src.data.data_generator import DataGenerator
from src.data.data_processor import DataProcessor
from src.models.model_factory import ModelFactory
from src.models.model_selector import ModelSelector
from src.features.protocol_specs import get_protocol_spec, get_all_protocols
from .model_trainer import ModelTrainer
from .incremental_trainer import IncrementalTrainer
from .model_evaluator import ModelEvaluator
from .feedback_optimizer import FeedbackOptimizer
from .automl_trainer import AutoMLTrainer

class SelfDrivingLoop:
    """自驱动自学习闭环系统"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """初始化自驱动学习系统"""
        self.config = config or ConfigManager()
        self.logger = get_logger("training.self_driving_loop")
        
        # 初始化各个组件
        self.data_storage = DataStorage(config=self.config)
        self.model_factory = ModelFactory(config=self.config)
        self.model_selector = ModelSelector(config=self.config)
        self.data_processor = DataProcessor(config=self.config)
        
        # 初始化训练器组件
        self.evaluator = ModelEvaluator()
        self.feedback_optimizer = FeedbackOptimizer(config=self.config)
        self.base_trainer = ModelTrainer(
            model_factory=self.model_factory,
            config=self.config,
            evaluator=self.evaluator
        )
        self.incremental_trainer = IncrementalTrainer(
            model_factory=self.model_factory,
            config=self.config,
            evaluator=self.evaluator
        )
        self.automl_trainer = AutoMLTrainer(
            config=self.config,
            data_storage=self.data_storage,
            model_factory=self.model_factory,
            model_selector=self.model_selector
        )
        
        # 获取自驱动学习配置
        self_driving_config = self.config.get_self_driving_config()
        
        # 配置参数
        self.loop_config = {
            "data_check_interval": self_driving_config.get("data_check_interval", 3600),  # 1小时
            "min_new_samples": self_driving_config.get("min_new_samples", 1000),
            "max_history_days": self_driving_config.get("max_history_days", 30),
            "evaluation_threshold": self_driving_config.get("evaluation_threshold", 0.75),
            "retrain_interval": self_driving_config.get("retrain_interval", 86400 * 3),  # 3天
            "data_sources": self_driving_config.get("data_sources", ["file", "generated"])
        }
        
        # 状态变量
        self._is_running = False
        self._last_data_check_time = 0
        self._last_full_train_time = {}
        self._new_data_count = {}
        
        self.logger.info("自驱动自学习闭环系统初始化完成")
    
    def start_loop(self) -> None:
        """启动自驱动学习循环"""
        if self._is_running:
            self.logger.warning("自驱动学习循环已在运行中")
            return
            
        self._is_running = True
        self.logger.info("启动自驱动学习循环")
        
        try:
            while self._is_running:
                current_time = time.time()
                
                # 检查是否需要检查新数据
                if current_time - self._last_data_check_time >= self.loop_config["data_check_interval"]:
                    self._last_data_check_time = current_time
                    self._execute_learning_cycle()
                
                # 等待下一次检查
                time.sleep(60)  # 每分钟检查一次停止信号
                
        except Exception as e:
            self.logger.error(f"自驱动学习循环出错: {str(e)}", exc_info=True)
        finally:
            self._is_running = False
            self.logger.info("自驱动学习循环已停止")
    
    def stop_loop(self) -> None:
        """停止自驱动学习循环"""
        self._is_running = False
        self.logger.info("正在停止自驱动学习循环...")
    
    def _execute_learning_cycle(self) -> None:
        """执行一个完整的学习周期"""
        self.logger.info("开始执行学习周期")
        
        try:
            # 1. 数据采集/生成
            X, y, protocol_labels, feature_names = self._collect_or_generate_data()
            
            if len(X) == 0:
                self.logger.warning("没有可用的训练数据，跳过本轮训练")
                return
            
            # 2. 模型选择
            model_type = self._select_best_model(X, y, protocol_labels)
            
            # 3. 模型训练
            model, metrics, model_path = self._train_model(model_type, X, y, protocol_labels)
            
            # 4. 模型评估
            evaluation_result = self._evaluate_model(model, X, y, feature_names)
            
            # 5. 反馈优化（传递训练数据用于可能的模型更换）
            optimization_result = self._optimize_based_on_feedback(
                model=model, 
                model_type=model_type, 
                evaluation_result=evaluation_result, 
                protocol_labels=protocol_labels, 
                X=X, 
                y=y
            )
            
            # 6. 决策是否需要再训练
            if self._should_retrain(evaluation_result):
                self.logger.info("根据评估结果，需要进行再训练")
                self._retrain_model(model_type, X, y, protocol_labels)
            
            self.logger.info("学习周期执行完成")
            
        except Exception as e:
            self.logger.error(f"执行学习周期时出错: {str(e)}", exc_info=True)
    
    def _collect_or_generate_data(self) -> Tuple[np.ndarray, np.ndarray, Optional[List[int]], List[str]]:
        """采集或生成训练数据"""
        self.logger.info("开始采集或生成训练数据")
        
        try:
            # 根据配置的数据源优先级依次尝试
            data_sources = self.loop_config.get("data_sources", ["file", "generated"])
            
            for source in data_sources:
                if source == "file":
                    # 尝试从文件加载数据
                    result = self._load_data_from_files()
                    if len(result[0]) > 0:  # 如果成功加载数据
                        self.logger.info("成功从文件加载数据")
                        return result
                elif source == "generated":
                    # 生成模拟数据
                    result = self._generate_simulated_data()
                    if len(result[0]) > 0:  # 如果成功生成数据
                        self.logger.info("成功生成模拟数据")
                        return result
                elif source == "capture":
                    # 尝试从网络捕获数据
                    result = self._capture_network_data()
                    if len(result[0]) > 0:  # 如果成功捕获数据
                        self.logger.info("成功捕获网络数据")
                        return result
                elif source == "pcap":
                    # 尝试从pcap文件加载数据
                    result = self._load_pcap_data()
                    if len(result[0]) > 0:  # 如果成功加载数据
                        self.logger.info("成功从pcap文件加载数据")
                        return result
            
            self.logger.warning("所有数据源都未能提供有效数据")
            return np.array([]), np.array([]), None, []
            
        except Exception as e:
            self.logger.error(f"数据采集/生成失败: {str(e)}", exc_info=True)
            return np.array([]), np.array([]), None, []
    
    def _load_data_from_files(self) -> Tuple[np.ndarray, np.ndarray, Optional[List[int]], List[str]]:
        """从文件加载数据"""
        try:
            # 检查是否有现成的训练数据
            processed_data_dir = self.config.get("data.processed_dir", "data/processed")
            
            if os.path.exists(processed_data_dir) and any(f.endswith('.csv') for f in os.listdir(processed_data_dir)):
                # 使用现有数据
                self.logger.info(f"从 {processed_data_dir} 加载现有数据")
                X, y = self.data_processor.load_processed_data(processed_data_dir)
                
                # 预处理数据
                model_compatible_features, _ = self.data_processor.get_model_compatible_features()
                available_features = [f for f in model_compatible_features if f in X.columns]
                X_filtered = X[available_features]
                X_processed = self.data_processor.preprocess_features(X_filtered, fit=True)
                
                # 生成协议标签（如果存在协议列）
                protocol_labels = None
                if 'protocol' in X.columns:
                    protocol_labels = X['protocol'].tolist()
                elif 'protocol_num' in X.columns:
                    protocol_labels = X['protocol_num'].tolist()
                else:
                    protocol_labels = [6] * len(X)  # 默认TCP协议
                
                feature_names = available_features
                
                self.logger.info(f"从文件加载数据完成，共 {len(X_processed)} 个样本")
                return self._to_numpy_arrays(X_processed, y, protocol_labels, feature_names)
            else:
                self.logger.info("未找到现有数据文件")
                return np.array([]), np.array([]), None, []
                
        except Exception as e:
            self.logger.error(f"从文件加载数据失败: {str(e)}", exc_info=True)
            return np.array([]), np.array([]), None, []
    
    def _generate_simulated_data(self) -> Tuple[np.ndarray, np.ndarray, Optional[List[int]], List[str]]:
        """生成模拟数据"""
        try:
            self.logger.info("生成模拟训练数据")
            processed_data_dir = self.config.get("data.processed_dir", "data/processed")
            
            # 生成模拟数据
            data_generator = DataGenerator()
            data_generator.generate(num_samples=5000, output_dir=processed_data_dir)
            X, y = self.data_processor.load_processed_data(processed_data_dir)
            
            # 预处理数据
            model_compatible_features, _ = self.data_processor.get_model_compatible_features()
            available_features = [f for f in model_compatible_features if f in X.columns]
            X_filtered = X[available_features]
            X_processed = self.data_processor.preprocess_features(X_filtered, fit=True)
            
            # 生成协议标签（模拟）
            protocol_labels = [6] * len(X)  # 默认TCP协议
            feature_names = available_features
            
            self.logger.info(f"模拟数据生成完成，共 {len(X_processed)} 个样本")
            return self._to_numpy_arrays(X_processed, y, protocol_labels, feature_names)
            
        except Exception as e:
            self.logger.error(f"生成模拟数据失败: {str(e)}", exc_info=True)
            return np.array([]), np.array([]), None, []
    
    def _capture_network_data(self) -> Tuple[np.ndarray, np.ndarray, Optional[List[int]], List[str]]:
        """从网络捕获数据"""
        try:
            self.logger.info("尝试从网络捕获数据")
            
            # 检查是否可以导入网络捕获相关模块
            try:
                from src.capture.packet_capture import PacketCapture
                from src.capture.session_tracker import SessionTracker
            except ImportError as e:
                self.logger.warning(f"无法导入网络捕获模块: {str(e)}")
                return np.array([]), np.array([]), None, []
            
            # 获取配置
            capture_config = self.config.get("capture", {})
            interface = capture_config.get("interface", "eth0")
            duration = capture_config.get("self_driving_capture_duration", 60)  # 默认捕获60秒
            
            self.logger.info(f"开始捕获网络数据，接口: {interface}, 持续时间: {duration}秒")
            
            # 初始化会话跟踪器和数据存储
            session_tracker = SessionTracker()
            session_tracker.start()
            
            # 创建数据存储目录
            capture_data_dir = self.config.get("data.capture_dir", "data/capture")
            os.makedirs(capture_data_dir, exist_ok=True)
            
            # 创建数据存储对象
            capture_storage = DataStorage(config=self.config)
            
            # 初始化数据处理器
            data_processor = DataProcessor(config=self.config)
            
            # 创建包捕获器
            packet_capture = PacketCapture(
                interface=interface
            )
            
            # 设置BPF过滤规则
            bpf_filter = capture_config.get("filter", "")
            if bpf_filter:
                packet_capture.set_filter(bpf_filter)
            
            # 启动捕获
            packet_capture.start()
            
            # 捕获指定时间的数据
            start_time = time.time()
            
            while time.time() - start_time < duration and self._is_running:
                # 获取下一个数据包
                packet_data = packet_capture.get_next_packet()
                if packet_data:
                    header, packet = packet_data
                    # 直接处理数据包（简化处理）
                    # 在实际应用中，这里需要更复杂的数据包解析逻辑
                    pass
                time.sleep(0.001)  # 短暂休眠避免CPU占用过高
            
            # 停止捕获
            packet_capture.stop()
            session_tracker.stop()
            
            self.logger.info("网络数据捕获完成")
            
            # 从会话跟踪器提取特征
            sessions = session_tracker.sessions
            if not sessions:
                self.logger.warning("未捕获到任何网络会话")
                return np.array([]), np.array([]), None, []
            
            # 将会话转换为特征数据
            features_data = []
            labels = []  # 这里需要根据实际情况确定标签
            
            for session_id, session in sessions.items():
                # 将会话转换为字典
                session_dict = session.to_dict()
                
                # 构造特征字典
                features = {
                    "packet_count": session_dict["total_packets"],
                    "byte_count": session_dict["total_bytes"],
                    "flow_duration": session_dict["duration"],
                    "avg_packet_size": session_dict["total_bytes"] / max(session_dict["total_packets"], 1),
                    "std_packet_size": 0,  # 简化处理
                    "min_packet_size": 0,  # 简化处理
                    "max_packet_size": 0,  # 简化处理
                    "bytes_per_second": session_dict["total_bytes"] / max(session_dict["duration"], 0.001),
                    "packets_per_second": session_dict["total_packets"] / max(session_dict["duration"], 0.001),
                    "tcp_syn_count": 0,  # 简化处理
                    "tcp_ack_count": 0,  # 简化处理
                    "tcp_fin_count": 0,  # 简化处理
                    "tcp_rst_count": 0,  # 简化处理
                    "tcp_flag_ratio": 0,  # 简化处理
                    "payload_entropy": 0  # 简化处理
                }
                
                features_data.append(features)
                # 简化处理：默认标记为正常流量（0）
                # 在实际应用中，这里需要更复杂的标签生成逻辑
                labels.append(0)
            
            if not features_data:
                self.logger.warning("未能从捕获的数据中提取有效特征")
                return np.array([]), np.array([]), None, []
            
            # 转换为DataFrame
            X_df = pd.DataFrame(features_data)
            y_series = pd.Series(labels)
            
            # 预处理数据
            model_compatible_features, _ = data_processor.get_model_compatible_features()
            available_features = [f for f in model_compatible_features if f in X_df.columns]
            X_filtered = X_df[available_features]
            X_processed = data_processor.preprocess_features(X_filtered, fit=True)
            
            # 生成协议标签
            protocol_labels = [6] * len(X_df)  # 简化处理，默认TCP协议
            feature_names = available_features
            
            self.logger.info(f"网络数据捕获完成，共 {len(X_processed)} 个样本")
            return self._to_numpy_arrays(X_processed, y_series, protocol_labels, feature_names)
            
        except Exception as e:
            self.logger.error(f"网络数据捕获失败: {str(e)}", exc_info=True)
            return np.array([]), np.array([]), None, []
    
    def _load_pcap_data(self) -> Tuple[np.ndarray, np.ndarray, Optional[List[int]], List[str]]:
        """从pcap文件加载数据"""
        try:
            self.logger.info("尝试从pcap文件加载数据")
            
            # 获取pcap文件目录配置
            pcap_dir = self.config.get("data.pcap_dir", "data/pcap")
            
            if not os.path.exists(pcap_dir):
                self.logger.warning(f"pcap目录不存在: {pcap_dir}")
                return np.array([]), np.array([]), None, []
            
            # 查找pcap文件
            pcap_files = [f for f in os.listdir(pcap_dir) if f.endswith('.pcap') or f.endswith('.pcapng')]
            if not pcap_files:
                self.logger.warning(f"pcap目录中未找到pcap文件: {pcap_dir}")
                return np.array([]), np.array([]), None, []
            
            # 选择最新的pcap文件
            latest_pcap = max(pcap_files, key=lambda f: os.path.getmtime(os.path.join(pcap_dir, f)))
            pcap_path = os.path.join(pcap_dir, latest_pcap)
            
            self.logger.info(f"处理pcap文件: {pcap_path}")
            
            # 尝试导入pcap处理相关模块
            try:
                from src.capture.packet_capture import PacketCapture
                from src.capture.session_tracker import SessionTracker
            except ImportError as e:
                self.logger.warning(f"无法导入pcap处理模块: {str(e)}")
                return np.array([]), np.array([]), None, []
            
            # 初始化数据处理器
            data_processor = DataProcessor(config=self.config)
            
            # 初始化会话跟踪器
            session_tracker = SessionTracker()
            session_tracker.start()
            
            # 初始化数据存储
            capture_storage = DataStorage(config=self.config)
            
            # 创建包捕获器（离线模式）
            packet_capture = PacketCapture(offline_file=pcap_path)
            
            # 启动捕获
            packet_capture.start()
            
            # 解析所有数据包
            packet_count = 0
            max_packets = 10000  # 限制处理的数据包数量以避免内存问题
            
            while self._is_running:
                # 获取下一个数据包
                packet_data = packet_capture.get_next_packet()
                if not packet_data:
                    break  # 没有更多数据包
                
                # 在实际应用中，这里需要解析数据包并将其传递给会话跟踪器
                # 由于缺少PacketParser类，我们简化处理
                packet_count += 1
                
                # 限制处理的数据包数量以避免内存问题
                if packet_count >= max_packets:
                    self.logger.info("达到数据包处理上限，停止处理")
                    break
                    
                # 短暂休眠避免CPU占用过高
                time.sleep(0.001)
            
            # 停止捕获和会话跟踪
            packet_capture.stop()
            session_tracker.stop()
            
            self.logger.info(f"处理了 {packet_count} 个数据包")
            
            # 从会话跟踪器提取特征
            sessions = session_tracker.sessions
            if not sessions:
                self.logger.warning("pcap文件中未发现有效会话")
                return np.array([]), np.array([]), None, []
            
            # 将会话转换为特征数据
            features_data = []
            labels = []  # 这里需要根据实际情况确定标签
            
            for session_id, session in sessions.items():
                # 将会话转换为字典
                session_dict = session.to_dict()
                
                # 构造特征字典
                features = {
                    "packet_count": session_dict["total_packets"],
                    "byte_count": session_dict["total_bytes"],
                    "flow_duration": session_dict["duration"],
                    "avg_packet_size": session_dict["total_bytes"] / max(session_dict["total_packets"], 1),
                    "std_packet_size": 0,  # 简化处理
                    "min_packet_size": 0,  # 简化处理
                    "max_packet_size": 0,  # 简化处理
                    "bytes_per_second": session_dict["total_bytes"] / max(session_dict["duration"], 0.001),
                    "packets_per_second": session_dict["total_packets"] / max(session_dict["duration"], 0.001),
                    "tcp_syn_count": 0,  # 简化处理
                    "tcp_ack_count": 0,  # 简化处理
                    "tcp_fin_count": 0,  # 简化处理
                    "tcp_rst_count": 0,  # 简化处理
                    "tcp_flag_ratio": 0,  # 简化处理
                    "payload_entropy": 0  # 简化处理
                }
                
                features_data.append(features)
                # 简化处理：默认标记为正常流量（0）
                # 在实际应用中，这里需要更复杂的标签生成逻辑
                labels.append(0)
            
            if not features_data:
                self.logger.warning("未能从pcap数据中提取有效特征")
                return np.array([]), np.array([]), None, []
            
            # 转换为DataFrame
            X_df = pd.DataFrame(features_data)
            y_series = pd.Series(labels)
            
            # 预处理数据
            model_compatible_features, _ = data_processor.get_model_compatible_features()
            available_features = [f for f in model_compatible_features if f in X_df.columns]
            X_filtered = X_df[available_features]
            X_processed = data_processor.preprocess_features(X_filtered, fit=True)
            
            # 生成协议标签
            protocol_labels = [6] * len(X_df)  # 简化处理，默认TCP协议
            feature_names = available_features
            
            self.logger.info(f"pcap数据处理完成，共 {len(X_processed)} 个样本")
            return self._to_numpy_arrays(X_processed, y_series, protocol_labels, feature_names)
            
        except Exception as e:
            self.logger.error(f"pcap数据加载失败: {str(e)}", exc_info=True)
            return np.array([]), np.array([]), None, []
    
    def _to_numpy_arrays(self, X, y, protocol_labels, feature_names) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
        """将数据转换为NumPy数组格式"""
        X_array = X.values if hasattr(X, 'values') else np.array(X)
        y_array = y.values if hasattr(y, 'values') else np.array(y)
        return X_array, y_array, protocol_labels, feature_names
    
    def _select_best_model(self, X: np.ndarray, y: np.ndarray, protocol_labels: Optional[List[int]]) -> str:
        """选择最佳模型类型"""
        self.logger.info("开始选择最佳模型")
        
        try:
            # 简化实现：基于数据特征选择模型
            sample_count = len(X)
            feature_count = X.shape[1] if len(X.shape) > 1 else 0
            
            # 根据样本数量和特征数量选择模型
            if sample_count > 10000 and feature_count > 20:
                model_type = "xgboost"
            elif sample_count > 5000:
                model_type = "random_forest"
            else:
                model_type = "mlp"
            
            self.logger.info(f"选择模型类型: {model_type}")
            return model_type
            
        except Exception as e:
            self.logger.error(f"模型选择失败: {str(e)}", exc_info=True)
            return "xgboost"  # 默认模型
    
    def _train_model(self, model_type: str, X: np.ndarray, y: np.ndarray, protocol_labels: Optional[List[int]]) -> Tuple[Any, Dict[str, float], str]:
        """训练模型"""
        self.logger.info(f"开始训练 {model_type} 模型")
        
        try:
            model, metrics, model_path = self.base_trainer.train_new_model(
                model_type=model_type,
                X=X,
                y=y,
                protocol_labels=protocol_labels
            )
            
            self.logger.info(f"模型训练完成，F1分数: {metrics.get('f1', 0):.4f}")
            return model, metrics, model_path
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}", exc_info=True)
            raise
    
    def _evaluate_model(self, model, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """评估模型"""
        self.logger.info("开始评估模型")
        
        try:
            metrics = self.evaluator.evaluate_model(
                model=model,
                X_test=X,
                y_test=y,
                feature_names=feature_names
            )
            
            self.logger.info(f"模型评估完成，F1分数: {metrics.get('f1', 0):.4f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"模型评估失败: {str(e)}", exc_info=True)
            # 返回默认评估结果
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    
    def _optimize_threshold(self, model, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        优化分类阈值以最大化F1分数
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试标签
            
        Returns:
            最佳阈值
        """
        try:
            # 获取预测概率
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # 计算不同阈值下的F1分数
            thresholds = np.arange(0.1, 0.9, 0.05)
            best_threshold = 0.5
            best_f1 = 0.0
            
            from sklearn.metrics import f1_score
            for threshold in thresholds:
                y_pred = (y_proba >= threshold).astype(int)
                f1 = f1_score(y_test, y_pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            self.logger.info(f"阈值优化完成 - 最佳阈值: {best_threshold:.3f}, F1分数: {best_f1:.4f}")
            return best_threshold
            
        except Exception as e:
            self.logger.warning(f"阈值优化失败: {str(e)}，使用默认阈值0.5")
            return 0.5

    def _optimize_based_on_feedback(
        self, 
        model, 
        model_type: str, 
        evaluation_result: Dict[str, Any], 
        protocol_labels: Optional[List[int]], 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Dict[str, Any]:
        """基于反馈进行优化"""
        self.logger.info("开始基于反馈进行优化")
        
        try:
            # 获取特征重要性
            feature_importance = None
            if hasattr(model, "get_feature_importance"):
                try:
                    feature_importance = model.get_feature_importance()
                except Exception as e:
                    self.logger.warning(f"获取特征重要性时出错: {str(e)}")
            
            # 执行优化
            optimization_result = self.feedback_optimizer.optimize_based_on_evaluation(
                model_type=model_type,
                evaluation_metrics=evaluation_result,
                protocol=protocol_labels[0] if protocol_labels else None,
                feature_importance=feature_importance,
                model_factory=self.model_factory
            )
            
            # 自动执行优化建议（包括模型更换）
            execution_result = self.feedback_optimizer.auto_execute_recommendations(
                recommendations=optimization_result,
                X=X,
                y=y,
                protocol_labels=protocol_labels,
                model_factory=self.model_factory
            )
            
            if execution_result["model_changed"]:
                self.logger.info(f"自动模型更换完成: {model_type} -> {execution_result['new_model'].__class__.__name__}")
                self.logger.info(f"新模型F1分数: {execution_result['metrics'].get('f1', 0):.4f}")
            
            # 保存优化历史
            self.feedback_optimizer.save_optimization_history()
            
            self.logger.info("反馈优化完成")
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"反馈优化失败: {str(e)}", exc_info=True)
            return {}

    def _should_retrain(self, evaluation_result: Dict[str, Any]) -> bool:
        """判断是否需要再训练"""
        f1_score = evaluation_result.get("f1", 0)
        return f1_score < self.loop_config["evaluation_threshold"]
    
    def _retrain_model(self, model_type: str, X: np.ndarray, y: np.ndarray, protocol_labels: Optional[List[int]]) -> None:
        """再训练模型"""
        self.logger.info(f"开始再训练 {model_type} 模型")
        
        try:
            # 使用AutoML进行再训练
            self.automl_trainer._train_single_model(
                model_type=model_type,
                X=X,
                y=y,
                protocol=protocol_labels[0] if protocol_labels else None,
                feature_names=[f"feature_{i}" for i in range(X.shape[1])] if len(X.shape) > 1 else ["feature_0"]
            )
            
            self.logger.info("模型再训练完成")
            
        except Exception as e:
            self.logger.error(f"模型再训练失败: {str(e)}", exc_info=True)