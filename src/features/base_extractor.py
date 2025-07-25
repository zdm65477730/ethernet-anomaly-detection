import time
import threading
from abc import ABC, abstractmethod
from src.system.base_component import BaseComponent
from src.utils.logger import get_logger

class BaseFeatureExtractor(BaseComponent, ABC):
    """
    特征提取器基类，定义特征提取的接口规范
    
    所有特征提取器都应继承此类并实现抽象方法
    """
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger(self.__class__.__name__)
        
        # 特征配置
        self.enabled_features = []  # 启用的特征列表
        self.disabled_features = []  # 禁用的特征列表
        
        # 特征元数据
        self.feature_metadata = {}  # {feature_name: {type: "numeric"|"categorical", description: "...", ...}}
        
        # 特征统计信息
        self.feature_stats = {}  # {feature_name: {min: x, max: y, mean: z, ...}}
        
        # 线程相关
        self._processing_thread = None
        self._feature_queue = None
    
    @abstractmethod
    def extract_features(self, packet, session):
        """
        从单个数据包和会话中提取特征
        
        参数:
            packet: 数据包对象，包含解析后的协议信息
            session: 会话对象，包含该会话的上下文信息
            
        返回:
            特征字典 {feature_name: value}
        """
        pass
    
    @abstractmethod
    def extract_features_from_session(self, session):
        """
        从整个会话中提取特征
        
        参数:
            session: 会话对象，包含该会话的所有数据包和元数据
            
        返回:
            特征字典 {feature_name: value}
        """
        pass
    
    def enable_features(self, features):
        """
        启用指定的特征
        
        参数:
            features: 特征名称列表
        """
        if not isinstance(features, list):
            features = [features]
            
        for feature in features:
            if feature in self.disabled_features:
                self.disabled_features.remove(feature)
            if feature not in self.enabled_features:
                self.enabled_features.append(feature)
                self.logger.debug(f"已启用特征: {feature}")
    
    def disable_features(self, features):
        """
        禁用指定的特征
        
        参数:
            features: 特征名称列表
        """
        if not isinstance(features, list):
            features = [features]
            
        for feature in features:
            if feature in self.enabled_features:
                self.enabled_features.remove(feature)
            if feature not in self.disabled_features:
                self.disabled_features.append(feature)
                self.logger.debug(f"已禁用特征: {feature}")
    
    def is_feature_enabled(self, feature_name):
        """
        检查特征是否启用
        
        参数:
            feature_name: 特征名称
            
        返回:
            布尔值，True表示启用，False表示禁用
        """
        # 如果没有显式设置启用列表，则默认所有特征都启用（除了被禁用的）
        if not self.enabled_features:
            return feature_name not in self.disabled_features
        return feature_name in self.enabled_features
    
    def get_enabled_features(self):
        """获取所有启用的特征列表"""
        if not self.enabled_features:
            # 返回所有已知特征中未被禁用的
            return [f for f in self.feature_metadata.keys() if f not in self.disabled_features]
        return self.enabled_features.copy()
    
    def update_feature_performance(self, performance_data):
        """
        更新特征性能数据，用于特征选择和优化
        
        参数:
            performance_data: 特征性能字典 {feature_name: {tp: int, fp: int, ...}}
        """
        # 可以根据特征性能数据自动启用/禁用特征
        # 例如：禁用假阳性率过高的特征
        for feature, stats in performance_data.items():
            if feature not in self.feature_metadata:
                continue
                
            # 计算假阳性率
            fp = stats.get("fp", 0)
            tn = stats.get("tn", 0)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # 如果假阳性率过高，禁用该特征
            if fpr > 0.8 and stats.get("total", 0) > 50:  # 至少有50个样本
                self.disable_features(feature)
                self.logger.info(f"因假阳性率过高({fpr:.2f})自动禁用特征: {feature}")
            # 如果特征性能良好且被禁用，启用该特征
            elif fpr < 0.2 and stats.get("f1", 0) > 0.7 and feature in self.disabled_features:
                self.enable_features(feature)
                self.logger.info(f"因性能良好(F1: {stats.get('f1'):.2f})自动启用特征: {feature}")
    
    def update_feature_importance(self, importance_data):
        """
        更新特征重要性数据，用于特征选择
        
        参数:
            importance_data: 特征重要性字典 {feature_name: importance_score}
        """
        # 可以根据重要性自动启用/禁用特征
        for feature, score in importance_data.items():
            # 禁用重要性极低的特征
            if score < 0.001 and self.is_feature_enabled(feature):
                self.disable_features(feature)
                self.logger.info(f"因重要性极低({score:.4f})自动禁用特征: {feature}")
            # 启用重要性高但被禁用的特征
            elif score > 0.05 and not self.is_feature_enabled(feature):
                self.enable_features(feature)
                self.logger.info(f"因重要性高({score:.4f})自动启用特征: {feature}")
    
    def start(self):
        """启动特征提取器"""
        if self._is_running:
            self.logger.warning(f"{self.__class__.__name__}已在运行中")
            return
            
        super().start()
        self.logger.info(f"{self.__class__.__name__}已启动")
    
    def stop(self):
        """停止特征提取器"""
        if not self._is_running:
            return
            
        super().stop()
        self.logger.info(f"{self.__class__.__name__}已停止")
    
    def get_status(self):
        """获取特征提取器状态"""
        status = super().get_status()
        status.update({
            "enabled_features_count": len(self.get_enabled_features()),
            "disabled_features_count": len(self.disabled_features),
            "total_known_features": len(self.feature_metadata)
        })
        return status
