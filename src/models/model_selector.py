import os
import json
import time
import numpy as np
from typing import Optional, Dict, Any
from src.utils.logger import get_logger
from src.config.config_manager import ConfigManager
from src.features.protocol_specs import get_protocol_spec, get_protocol_number
from src.system.base_component import BaseComponent

class ModelSelector(BaseComponent):
    """模型选择器，根据协议类型和历史性能选择最优模型"""
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ConfigManager()
        self.logger = get_logger("model_selector")
        
        # 模型性能历史记录路径
        self.models_dir = self.config.get("model.models_dir", "models")
        self.performance_history_path = os.path.join(
            "config", "protocol_model_performance.json"
        )
        
        # 确保模型目录存在
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 加载性能历史记录
        self.performance_history = self._load_performance_history()
        
        # 缓存最优模型选择结果
        self._best_model_cache = {}
        self._cache_expiry = 300  # 缓存过期时间(秒)
        self._cache_update_time = time.time()
        
        # 新增属性用于基于特征选择模型
        self.model_weights = {}
        self.selection_history = []
        self.feature_importance = {}
    
    def get_status(self) -> Dict[str, any]:
        """
        获取组件状态信息
        
        返回:
            包含组件状态的字典
        """
        status = super().get_status()
        status.update({
            "performance_history_size": len(self.performance_history),
            "cache_size": len(self._best_model_cache),
            "models_dir": self.models_dir
        })
        return status
    
    def start(self) -> None:
        """启动模型选择器"""
        if self.is_running:
            self.logger.warning("模型选择器已在运行中")
            return
            
        super().start()
        self.logger.info("模型选择器已启动")
    
    def stop(self) -> None:
        """停止模型选择器"""
        if not self.is_running:
            return
            
        # 保存性能历史记录
        self._save_performance_history()
        
        super().stop()
        self.logger.info("模型选择器已停止")
    
    def _load_performance_history(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        加载模型性能历史记录
        
        格式: {
            "tcp": {
                "xgboost": {"f1": 0.92, "precision": 0.91, "recall": 0.93, "timestamp": 1620000000},
                "lstm": {"f1": 0.94, "precision": 0.93, "recall": 0.95, "timestamp": 1620000000}
            },
            ...
        }
        """
        try:
            if os.path.exists(self.performance_history_path):
                with open(self.performance_history_path, "r") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.error(f"加载模型性能历史失败: {str(e)}")
            return {}
    
    def _save_performance_history(self) -> None:
        """保存模型性能历史记录到文件"""
        try:
            # 确保所有数值都是JSON可序列化的类型
            def convert_to_serializable(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_to_serializable(value) for key, value in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_serializable(item) for item in obj]
                else:
                    return obj
            
            serializable_history = convert_to_serializable(self.performance_history)
            
            with open(self.performance_history_path, "w") as f:
                json.dump(serializable_history, f, indent=2)
            self.logger.debug(f"已保存模型性能历史到 {self.performance_history_path}")
        except Exception as e:
            self.logger.error(f"保存模型性能历史失败: {str(e)}")
    
    def update_performance(
        self,
        protocol,
        model_type: str,
        metrics: Dict[str, float],
        sample_count: Optional[int] = None
    ) -> None:
        """
        更新模型性能记录
        
        Args:
            protocol: 协议类型（整数协议号或字符串协议名）
            model_type: 模型类型
            metrics: 性能指标字典
            sample_count: 样本数量
        """
        try:
            # 标准化协议名称
            if isinstance(protocol, (int, float, np.integer, np.floating)):
                # 处理数值型协议号，包括numpy类型
                if np.isnan(protocol):
                    protocol_name = "unknown"
                else:
                    proto_spec = get_protocol_spec(int(protocol))
                    protocol_name = proto_spec["name"]
            else:
                # 处理字符串协议名
                protocol_name = str(protocol).lower()
                if not protocol_name or protocol_name == "nan":
                    protocol_name = "unknown"
            
            # 确保协议在性能历史中存在
            if protocol_name not in self.performance_history:
                self.performance_history[protocol_name] = {}
            
            # 更新性能记录
            if model_type not in self.performance_history[protocol_name]:
                self.performance_history[protocol_name][model_type] = {}
            
            # 更新指标
            self.performance_history[protocol_name][model_type].update(metrics)
            
            # 添加样本数量
            if sample_count is not None:
                self.performance_history[protocol_name][model_type]["sample_count"] = sample_count
            
            # 保存性能历史
            self._save_performance_history()
            
            self.logger.debug(f"已更新协议 {protocol_name} 的 {model_type} 模型性能记录")
            
        except Exception as e:
            self.logger.error(f"更新性能记录时出错: {str(e)}", exc_info=True)
        
        # 清除缓存
        self._invalidate_cache()
        
        self.logger.debug(f"已更新 {protocol_name} 协议的 {model_type} 模型性能记录")
    
    def select_best_model(self, protocol, candidates=None):
        """
        为指定协议选择最佳模型类型
        
        Args:
            protocol: 协议类型（整数协议号或字符串协议名）
            candidates: 候选模型列表
            
        Returns:
            最佳模型类型字符串
        """
        # 检查缓存
        current_time = time.time()
        if (current_time - self._cache_update_time) < self._cache_expiry:
            cache_key = str(protocol)
            if cache_key in self._best_model_cache:
                return self._best_model_cache[cache_key]
        
        # 标准化协议名称
        if isinstance(protocol, (int, float, np.integer, np.floating)):
            # 处理数值型协议号，包括numpy类型
            # 特别处理NaN值
            if np.isnan(protocol):
                protocol_name = "unknown"
                proto_spec = get_protocol_spec(None)  # 获取默认协议规范
            else:
                proto_spec = get_protocol_spec(int(protocol))
                protocol_name = proto_spec["name"]
            default_candidates = proto_spec["model_preference"]
        else:
            # 处理字符串协议名
            protocol_name = str(protocol).lower()
            # 处理空字符串或None
            if not protocol_name or protocol_name == "nan":
                protocol_name = "unknown"
                proto_spec = get_protocol_spec(None)
            else:
                proto_spec = get_protocol_spec(get_protocol_number(protocol_name))
            default_candidates = proto_spec["model_preference"]
        
        # 确定候选模型
        model_candidates = candidates or default_candidates
        
        # 如果没有性能记录，返回默认首选模型
        if protocol_name not in self.performance_history:
            best_model = model_candidates[0]
            self._best_model_cache[str(protocol)] = best_model
            self._cache_update_time = current_time
            self.logger.debug(f"协议 {protocol_name} 无历史性能记录，使用默认模型: {best_model}")
            return best_model
        
        # 根据历史性能选择最佳模型
        protocol_history = self.performance_history[protocol_name]
        best_model = None
        best_score = -1
        
        for model_type in model_candidates:
            if model_type in protocol_history:
                # 使用F1分数作为选择标准
                metrics = protocol_history[model_type]
                f1_score = metrics.get("f1", 0)
                # 考虑样本数量的置信度
                sample_count = metrics.get("sample_count", 0)
                confidence = min(sample_count / 1000, 1.0)  # 1000样本以上置信度为1
                
                # 计算加权分数
                weighted_score = f1_score * confidence
                
                if weighted_score > best_score:
                    best_score = weighted_score
                    best_model = model_type
        
        # 如果没有找到合适的模型或分数太低，使用默认模型
        if not best_model or best_score < 0.1:
            best_model = model_candidates[0]
            self.logger.debug(f"协议 {protocol_name} 历史性能不佳，使用默认模型: {best_model}")
        else:
            self.logger.debug(f"协议 {protocol_name} 选择模型: {best_model} (得分: {best_score:.3f})")
        
        # 更新缓存
        self._best_model_cache[str(protocol)] = best_model
        self._cache_update_time = current_time
        
        return best_model
    
    def get_model_performance(
        self, 
        protocol: str or int, 
        model_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取指定协议和模型的性能指标"""
        # 标准化协议名称
        if isinstance(protocol, int):
            proto_spec = get_protocol_spec(protocol)
            protocol_name = proto_spec["name"]
        else:
            protocol_name = protocol.lower()
        
        # 检查协议是否有性能记录
        if protocol_name not in self.performance_history:
            return {}
        
        # 如果指定了模型类型，返回该模型的性能
        if model_type:
            return self.performance_history[protocol_name].get(model_type, {})
        
        # 否则返回所有模型的性能
        return self.performance_history[protocol_name]
    
    def _update_cache(self, protocol_key: str, model_type: str) -> None:
        """更新缓存"""
        self._best_model_cache[protocol_key] = model_type
        self._cache_update_time = time.time()
    
    def _invalidate_cache(self) -> None:
        """使缓存失效"""
        self._best_model_cache.clear()
        self._cache_update_time = 0
    
    def clear_history(self, protocol: Optional[str] = None) -> None:
        """
        清除性能历史记录
        
        参数:
            protocol: 协议名称，为None则清除所有
        """
        if protocol:
            protocol_name = protocol.lower()
            if protocol_name in self.performance_history:
                del self.performance_history[protocol_name]
                self._save_performance_history()
                self.logger.info(f"已清除 {protocol_name} 的模型性能历史")
        else:
            self.performance_history = {}
            self._save_performance_history()
            self.logger.info("已清除所有模型性能历史")
        
        # 清除缓存
        self._invalidate_cache()
