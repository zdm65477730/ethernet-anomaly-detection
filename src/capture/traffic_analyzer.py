import time
import threading
import logging
from queue import Queue, Empty
from typing import Optional, Dict, Any
from src.system.base_component import BaseComponent
from src.utils.logger import get_logger
from src.features.stat_extractor import StatFeatureExtractor
from src.features.temporal_extractor import TemporalFeatureExtractor
from src.config.config_manager import ConfigManager

class TrafficAnalyzer(BaseComponent):
    """流量分析器，负责从会话中提取特征并传递给检测模块"""
    
    def __init__(self, stat_extractor: Optional[StatFeatureExtractor] = None, 
                 temporal_extractor: Optional[TemporalFeatureExtractor] = None):
        super().__init__()
        self.logger = get_logger("traffic_analyzer")
        
        # 创建配置管理器实例并传递给特征提取器
        config = ConfigManager()
            
        self.stat_extractor = stat_extractor or StatFeatureExtractor(config=config)
        self.temporal_extractor = temporal_extractor or TemporalFeatureExtractor(config=config)
        
        self._analysis_thread = None  # 分析线程
        self._feature_queue = Queue(maxsize=1000)  # 特征队列，传递给检测模块
        
        # 滑动窗口配置
        self.window_size = 60  # 窗口大小(秒)
        self.window_step = 10  # 窗口步长(秒)

    def get_next_features(self, timeout=1):
        """获取下一组提取的特征"""
        try:
            return self._feature_queue.get(timeout=timeout)
        except Empty:
            return None

    def _analyze_session(self, session):
        """分析会话并提取特征"""
        try:
            if not session or not session.packets:
                return None
                
            # 提取统计特征
            stat_features = self.stat_extractor.extract_features_from_session(session)
            
            # 提取时序特征
            temporal_features = self.temporal_extractor.extract_features_from_session(
                session, window_size=self.window_size
            )
            
            # 合并特征
            features = {
                **stat_features,
                **temporal_features,
                "session_id": session.session_id,
                "protocol": session.protocol,
                "timestamp": time.time()
            }
            
            self.logger.debug(f"为会话 {session.session_id} 提取特征: {len(features)} 个特征")
            return features
            
        except Exception as e:
            self.logger.error(f"分析会话 {session.session_id if session else '未知'} 时出错: {str(e)}", exc_info=True)
            return None

    def _analysis_loop(self, session_tracker):
        """分析循环，处理更新的会话"""
        while self._is_running:
            try:
                # 获取更新的会话
                session = session_tracker.get_next_updated_session(timeout=1)
                if not session:
                    continue
                
                # 分析会话并提取特征
                features = self._analyze_session(session)
                if features and not self._feature_queue.full():
                    self._feature_queue.put(features)
                elif self._feature_queue.full():
                    self.logger.warning("特征队列已满，丢弃特征数据")
                    
            except Exception as e:
                self.logger.error(f"分析循环出错: {str(e)}", exc_info=True)
                time.sleep(1)

    def start(self, session_tracker=None):
        """启动流量分析器"""
        if self._is_running:
            self.logger.warning("流量分析器已在运行中")
            return
            
        if not session_tracker:
            raise ValueError("必须提供session_tracker参数")
            
        super().start()
        self._analysis_thread = threading.Thread(
            target=self._analysis_loop,
            args=(session_tracker,),
            daemon=True
        )
        self._analysis_thread.start()
        
        # 启动特征提取器
        self.stat_extractor.start()
        self.temporal_extractor.start()
        
        self.logger.info("流量分析器已启动")

    def stop(self):
        """停止流量分析器"""
        if not self._is_running:
            return
            
        super().stop()
        
        # 停止分析线程
        if self._analysis_thread and self._analysis_thread.is_alive():
            self._analysis_thread.join(timeout=5)
            if self._analysis_thread.is_alive():
                self.logger.warning("分析线程未能正常终止")
        
        # 停止特征提取器
        try:
            self.temporal_extractor.stop()
        except Exception as e:
            self.logger.error(f"停止时序特征提取器时出错: {str(e)}", exc_info=True)
            
        try:
            self.stat_extractor.stop()
        except Exception as e:
            self.logger.error(f"停止统计特征提取器时出错: {str(e)}", exc_info=True)
        
        self.logger.info("流量分析器已停止")

    def get_status(self):
        """获取组件状态"""
        status = super().get_status()
        status.update({
            "feature_queue_size": self._feature_queue.qsize(),
            "feature_queue_max_size": self._feature_queue.maxsize,
            "window_size": self.window_size,
            "window_step": self.window_step,
            "stat_extractor": self.stat_extractor.get_status(),
            "temporal_extractor": self.temporal_extractor.get_status()
        })
        return status
    