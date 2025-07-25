import os
import json
import time
from typing import Optional, Dict, Any
from src.utils.logger import get_logger
from src.config.config_manager import ConfigManager
from src.features.protocol_specs import get_protocol_spec, get_protocol_number

class ModelSelector:
    """模型选择器，根据协议类型和历史性能选择最优模型"""
    
    def __init__(self, config=None):
        self.logger = get_logger("model_selector")
        self.config = config or ConfigManager()
        
        # 模型性能历史记录路径
        self.models_dir = self.config.get("model.models_dir", "models")
        self.performance_history_path = os.path.join(
            self.models_dir, "protocol_model_performance.json"
        )
        
        # 确保模型目录存在
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 加载性能历史记录
        self.performance_history = self._load_performance_history()
        
        # 缓存最优模型选择结果
        self._best_model_cache = {}
        self._cache_expiry = 300  # 缓存过期时间(秒)
        self._cache_update_time = time.time()
    
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
            with open(self.performance_history_path, "w") as f:
                json.dump(self.performance_history, f, indent=2)
            self.logger.debug(f"已保存模型性能历史到 {self.performance_history_path}")
        except Exception as e:
            self.logger.error(f"保存模型性能历史失败: {str(e)}")
    
    def update_performance(
        self, 
        protocol: str or int, 
        model_type: str, 
        metrics: Dict[str, float]
    ) -> None:
        """
        更新协议-模型的性能记录
        
        参数:
            protocol: 协议名称或编号
            model_type: 模型类型
            metrics: 性能指标字典，应包含f1, precision, recall等
        """
        # 标准化协议名称
        if isinstance(protocol, int):
            proto_spec = get_protocol_spec(protocol)
            protocol_name = proto_spec["name"]
        else:
            protocol_name = protocol.lower()
            # 验证协议名称是否有效
            try:
                get_protocol_number(protocol_name)
            except ValueError:
                self.logger.warning(f"未知协议: {protocol_name}，使用默认处理")
        
        # 添加时间戳
        metrics_with_time = metrics.copy()
        metrics_with_time["timestamp"] = time.time()
        
        # 更新性能历史
        if protocol_name not in self.performance_history:
            self.performance_history[protocol_name] = {}
        
        self.performance_history[protocol_name][model_type] = metrics_with_time
        
        # 保存更新
        self._save_performance_history()
        
        # 清除缓存
        self._invalidate_cache()
        
        self.logger.debug(f"已更新 {protocol_name} 协议的 {model_type} 模型性能记录")
    
    def select_best_model(
        self, 
        protocol: str or int, 
        candidates: Optional[list] = None
    ) -> str:
        """
        为指定协议选择最优模型
        
        参数:
            protocol: 协议名称或编号
            candidates: 候选模型列表，为None则使用协议默认偏好
            
        返回:
            最优模型类型名称
        """
        # 检查缓存是否有效
        current_time = time.time()
        if (current_time - self._cache_update_time) < self._cache_expiry:
            cache_key = str(protocol)
            if cache_key in self._best_model_cache:
                return self._best_model_cache[cache_key]
        
        # 标准化协议名称
        if isinstance(protocol, int):
            proto_spec = get_protocol_spec(protocol)
            protocol_name = proto_spec["name"]
            default_candidates = proto_spec["model_preference"]
        else:
            protocol_name = protocol.lower()
            proto_spec = get_protocol_spec(get_protocol_number(protocol_name))
            default_candidates = proto_spec["model_preference"]
        
        # 确定候选模型
        model_candidates = candidates or default_candidates
        
        # 如果没有性能记录，返回默认首选模型
        if protocol_name not in self.performance_history:
            best_model = model_candidates[0]
            self.logger.debug(f"{protocol_name} 无性能记录，选择默认模型 {best_model}")
            self._update_cache(str(protocol), best_model)
            return best_model
        
        # 获取该协议的所有模型性能记录
        protocol_models = self.performance_history[protocol_name]
        
        # 过滤候选模型
        valid_models = [m for m in model_candidates if m in protocol_models]
        
        # 如果没有有效的候选模型性能记录，返回默认首选模型
        if not valid_models:
            best_model = model_candidates[0]
            self.logger.debug(f"{protocol_name} 无有效模型性能记录，选择默认模型 {best_model}")
            self._update_cache(str(protocol), best_model)
            return best_model
        
        # 按F1分数排序，选择最优模型
        # 对于相同分数，优先选择最新更新的模型
        def model_rating(model_type):
            metrics = protocol_models[model_type]
            return (metrics["f1"], metrics["timestamp"])
        
        valid_models.sort(key=model_rating, reverse=True)
        best_model = valid_models[0]
        
        self.logger.debug(
            f"为 {protocol_name} 选择最优模型 {best_model} "
            f"(F1: {protocol_models[best_model]['f1']:.4f})"
        )
        
        # 更新缓存
        self._update_cache(str(protocol), best_model)
        
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
    
    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        protocol_count = len(self.performance_history)
        total_model_entries = sum(len(models) for models in self.performance_history.values())
        
        return {
            "performance_history_path": self.performance_history_path,
            "protocol_count": protocol_count,
            "total_model_entries": total_model_entries,
            "cache_size": len(self._best_model_cache),
            "cache_valid": (time.time() - self._cache_update_time) < self._cache_expiry
        }
