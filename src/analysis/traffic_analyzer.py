import time
import threading
import logging
from queue import Queue, Empty
from typing import Dict, Any, Optional
from src.system.base_component import BaseComponent
from src.utils.logger import get_logger
from src.features.stat_extractor import StatFeatureExtractor
from src.features.temporal_extractor import TemporalFeatureExtractor
from src.features.protocol_specs import PROTOCOL_SPECS

class TrafficAnalyzer(BaseComponent):
    """流量分析器，负责从会话数据中提取特征"""
    
    def __init__(self, session_tracker, anomaly_detector, config: Optional[Dict[str, Any]] = None):
        """
        初始化流量分析器
        
        参数:
            session_tracker: 会话追踪器实例
            anomaly_detector: 异常检测器实例
            config: 配置字典
        """
        super().__init__()
        self.logger = get_logger("traffic_analyzer")
        
        # 初始化配置
        self.config = config or {}
        
        # 初始化特征提取器
        self.stat_extractor = StatFeatureExtractor(config)
        self.temporal_extractor = TemporalFeatureExtractor(config)
        
        # 会话数据队列（从session_tracker接收）
        self.session_queue = Queue(maxsize=1000)
        
        # 特征队列（发送给anomaly_detector）
        self.features_queue = Queue(maxsize=1000)
        
        # 分析线程
        self._analysis_thread = None
        
        # 保存组件引用
        self._session_tracker = session_tracker
        self._anomaly_detector = anomaly_detector
        
        self.logger.info("流量分析器初始化完成")
    
    def start(self):
        """启动流量分析器"""
        if self._is_running:
            self.logger.warning("流量分析器已在运行中")
            return True
            
        super().start()
        
        # 启动分析线程
        self._analysis_thread = threading.Thread(
            target=self._analysis_loop,
            daemon=True
        )
        self._analysis_thread.start()
        
        self.logger.info("流量分析器已启动")
        return True

    def enqueue_session(self, session_data: Dict[str, Any]):
        """
        将会话数据加入处理队列
        
        参数:
            session_data: 会话数据字典
        """
        try:
            if not self._is_running:
                self.logger.warning("流量分析器未运行，无法处理会话数据")
                return False
            
            # 将会话数据加入队列
            self.session_queue.put(session_data)
            self.logger.debug(f"会话数据已加入队列，当前队列大小: {self.session_queue.qsize()}")
            return True
            
        except Exception as e:
            self.logger.error(f"将会话数据加入队列时出错: {e}")
            return False

    def _analysis_loop(self):
        """分析循环"""
        while self._is_running:
            try:
                # 从队列获取会话数据
                session_data = self.session_queue.get(timeout=1)
                if not session_data:
                    continue
                
                self.logger.debug(f"开始分析会话: {session_data.get('session_id', 'unknown')}")
                
                # 提取特征
                features = self._extract_features(session_data)
                if features:
                    self.logger.debug(f"特征提取完成，特征数量: {len(features)}")
                    
                    # 将特征放入特征队列供异常检测器使用
                    if self.features_queue:
                        self.features_queue.put(features)
                        self.logger.debug("特征已放入队列")
                    else:
                        self.logger.warning("特征队列未设置")
                else:
                    self.logger.warning("特征提取失败")
                
                # 标记任务完成
                self.session_queue.task_done()
                
            except Empty:
                # 队列为空，继续循环
                continue
            except Exception as e:
                self.logger.error(f"分析循环出错: {e}", exc_info=True)
                time.sleep(1)

    def _extract_features(self, session_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        从会话数据中提取特征
        
        参数:
            session_data: 会话数据字典
            
        返回:
            提取的特征字典，如果失败则返回None
        """
        try:
            features = {}
            
            # 基本信息
            features['session_id'] = session_data.get('session_id', '')
            features['timestamp'] = session_data.get('end_time', time.time())
            features['src_ip'] = session_data.get('src_ip', '')
            features['dst_ip'] = session_data.get('dst_ip', '')
            features['src_port'] = session_data.get('src_port', 0)
            features['dst_port'] = session_data.get('dst_port', 0)
            features['protocol'] = session_data.get('protocol', 'unknown')
            
            # 流量特征
            features['flow_duration'] = session_data.get('flow_duration', 0)
            features['packet_count'] = session_data.get('packet_count', 0)
            features['byte_count'] = session_data.get('byte_count', 0)
            
            # 计算衍生特征
            if features['flow_duration'] > 0:
                features['bytes_per_second'] = features['byte_count'] / features['flow_duration']
                features['packets_per_second'] = features['packet_count'] / features['flow_duration']
            else:
                features['bytes_per_second'] = 0
                features['packets_per_second'] = 0
            
            # 包大小特征
            if features['packet_count'] > 0:
                features['avg_packet_size'] = features['byte_count'] / features['packet_count']
            else:
                features['avg_packet_size'] = 0
            
            self.logger.debug(f"提取特征完成: {features}")
            return features
            
        except Exception as e:
            self.logger.error(f"特征提取失败: {e}", exc_info=True)
            return None

    def stop(self):
        """停止流量分析器"""
        if not self._is_running:
            return True
            
        # 停止分析线程
        if self._analysis_thread and self._analysis_thread.is_alive():
            self._analysis_thread.join(timeout=5)
            if self._analysis_thread.is_alive():
                self.logger.warning("分析线程未能正常终止")
        
        super().stop()
        
        self.logger.info("流量分析器已停止")
        return True
    
    def get_status(self):
        """获取组件状态"""
        status = super().get_status()
        status.update({
            "session_queue_size": self.session_queue.qsize(),
            "features_queue_size": self.features_queue.qsize()
        })
        return status