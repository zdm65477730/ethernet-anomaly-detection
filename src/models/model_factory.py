import os
import re
import time
import pickle
from typing import Dict, Optional, List
from .base_model import BaseModel
from .traditional_models import (
    XGBoostModel,
    RandomForestModel,
    LogisticRegressionModel
)
from .deep_models import LSTMModel, MLPModel
from src.config.config_manager import ConfigManager
from src.utils.logger import get_logger
from src.system.base_component import BaseComponent

class ModelFactory(BaseComponent):
    """模型工厂，统一管理模型的创建、训练、保存和加载"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        super().__init__()
        self.config = config or ConfigManager()
        self.models_dir = self.config.get("model.models_dir", "models")
        self.logger = get_logger("model.factory")
        
        # 模型类型到类的映射
        self._model_classes = {
            "xgboost": XGBoostModel,
            "random_forest": RandomForestModel,
            "logistic_regression": LogisticRegressionModel,
            "lstm": LSTMModel,
            "mlp": MLPModel
        }
        
        # 已加载的模型实例缓存
        self._model_cache: Dict[str, BaseModel] = {}
        
        # 确保模型目录存在
        os.makedirs(self.models_dir, exist_ok=True)
    
    def get_status(self) -> Dict[str, any]:
        """
        获取组件状态信息
        
        返回:
            包含组件状态的字典
        """
        status = super().get_status()
        status.update({
            "model_cache_size": len(self._model_cache),
            "supported_models": list(self._model_classes.keys()),
            "models_dir": self.models_dir
        })
        return status
    
    def start(self) -> None:
        """启动模型工厂"""
        if self.is_running:
            self.logger.warning("模型工厂已在运行中")
            return
            
        super().start()
        self.logger.info("模型工厂已启动")
    
    def stop(self) -> None:
        """停止模型工厂"""
        if not self.is_running:
            return
            
        # 清空模型缓存
        self._model_cache.clear()
        
        super().stop()
        self.logger.info("模型工厂已停止")
    
    def set_latest_model(self, model_type: str, model_path: str) -> None:
        """
        设置最新模型路径
        
        参数:
            model_type: 模型类型
            model_path: 模型路径
        """
        # 创建软链接或记录最新模型路径
        latest_model_path = os.path.join(self.models_dir, model_type, "latest_model.pkl")
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(latest_model_path), exist_ok=True)
            
            # 如果链接已存在，先删除
            if os.path.exists(latest_model_path) or os.path.islink(latest_model_path):
                os.remove(latest_model_path)
            
            # 创建新链接
            os.symlink(os.path.abspath(model_path), latest_model_path)
            self.logger.debug(f"已更新 {model_type} 类型的最新模型链接: {latest_model_path}")
        except Exception as e:
            self.logger.warning(f"无法创建最新模型链接 {latest_model_path}: {e}")
    
    def create_model(self, model_type: str, **kwargs) -> BaseModel:
        """创建指定类型的模型"""
        if model_type not in self._model_classes:
            raise ValueError(f"不支持的模型类型: {model_type}")
            
        model_class = self._model_classes[model_type]
        # 确保传递model_type参数
        kwargs["model_type"] = model_type
        return model_class(**kwargs)
    
    def save_model(self, model: BaseModel, model_path: str) -> bool:
        """
        保存模型到文件
        
        参数:
            model: 要保存的模型实例
            model_path: 模型保存路径
            
        返回:
            是否保存成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # 保存模型
            model.save(model_path)
            
            self.logger.info(f"模型已保存至: {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"保存模型失败: {str(e)}")
            return False
    
    def load_model(self, model_type: str, model_path: str) -> BaseModel:
        """
        从文件加载模型
        
        参数:
            model_type: 模型类型
            model_path: 模型文件路径
            
        返回:
            加载的模型实例
            
        异常:
            FileNotFoundError: 模型文件不存在
            ValueError: 不支持的模型类型
        """
        if model_type not in self._model_classes:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        model_class = self._model_classes[model_type]
        model = model_class.load(model_path)
        
        self.logger.info(f"已从 {model_path} 加载 {model_type} 模型")
        return model
    
    def load_latest_model(self, model_type: str) -> BaseModel:
        """
        加载指定类型的最新模型
        
        参数:
            model_type: 模型类型
            
        返回:
            最新的模型实例
            
        异常:
            FileNotFoundError: 未找到模型文件
            ValueError: 不支持的模型类型
        """
        model_path = self.get_latest_model_path(model_type)
        if not model_path:
            raise FileNotFoundError(f"未找到 {model_type} 模型文件")
        
        return self.load_model(model_type, model_path)
    
    def get_latest_model_path(self, model_type: str) -> Optional[str]:
        """
        获取指定类型的最新模型路径
        
        参数:
            model_type: 模型类型
            
        返回:
            最新模型文件路径，未找到则返回None
        """
        if model_type not in self._model_classes:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 首先检查是否存在latest_model链接
        model_type_dir = os.path.join(self.models_dir, model_type)
        latest_model_link = os.path.join(model_type_dir, "latest_model.pkl")
        
        if os.path.exists(latest_model_link) or os.path.islink(latest_model_link):
            try:
                # 解析符号链接
                resolved_path = os.path.realpath(latest_model_link)
                if os.path.exists(resolved_path):
                    return resolved_path
            except Exception as e:
                self.logger.warning(f"解析符号链接失败 {latest_model_link}: {e}")
        
        # 如果没有符号链接或解析失败，查找最新的模型文件
        model_files = []
        # 也检查根models目录下的模型文件
        for root_dir in [model_type_dir, self.models_dir]:
            if not os.path.exists(root_dir):
                continue
                
            for f in os.listdir(root_dir):
                if f.endswith(".pkl") and f.startswith(f"{model_type}_"):
                    file_path = os.path.join(root_dir, f)
                    model_files.append((file_path, os.path.getmtime(file_path)))
        
        if not model_files:
            return None
        
        # 按修改时间排序，返回最新的
        model_files.sort(key=lambda x: x[1], reverse=True)
        return model_files[0][0]
    
    def list_models(self, model_type: Optional[str] = None) -> List[Dict]:
        """
        列出所有可用的模型
        
        参数:
            model_type: 模型类型，为None则列出所有类型
            
        返回:
            模型信息列表
        """
        models = []
        
        # 确定要搜索的模型类型
        model_types = [model_type] if model_type else list(self._model_classes.keys())
        
        for mt in model_types:
            if mt not in self._model_classes:
                continue
                
            model_type_dir = os.path.join(self.models_dir, mt)
            if not os.path.exists(model_type_dir):
                continue
            
            # 查找该类型的所有模型文件
            for f in os.listdir(model_type_dir):
                if not f.endswith(".pkl") or not f.startswith(f"{mt}_"):
                    continue
                
                file_path = os.path.join(model_type_dir, f)
                stat = os.stat(file_path)
                
                # 提取版本信息
                version_match = re.search(r"v(\d{8}_\d{6})", f)
                version = version_match.group(1) if version_match else "unknown"
                
                models.append({
                    "filename": f,
                    "path": file_path,
                    "version": version,
                    "size": stat.st_size,
                    "modified_time": stat.st_mtime
                })
        
        # 按修改时间排序
        models.sort(key=lambda x: x["modified_time"], reverse=True)
        return models
    
    def delete_model(self, model_type: str, version: str) -> bool:
        """
        删除指定版本的模型
        
        参数:
            model_type: 模型类型
            version: 模型版本
            
        返回:
            是否删除成功
        """
        if model_type not in self._model_classes:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        model_filename = f"{model_type}_v{version}.pkl"
        model_path = os.path.join(self.models_dir, model_type, model_filename)
        
        if not os.path.exists(model_path):
            self.logger.warning(f"模型文件不存在: {model_path}")
            return False
        
        try:
            os.remove(model_path)
            self.logger.info(f"已删除模型: {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"删除模型失败: {str(e)}")
            return False
    
    def get_model_feature_importance(self, model: BaseModel) -> Optional[Dict[str, float]]:
        """
        获取模型的特征重要性
        
        参数:
            model: 模型实例
            
        返回:
            特征重要性字典，特征名到重要性的映射
        """
        try:
            if hasattr(model, "get_feature_importance"):
                return model.get_feature_importance()
            else:
                self.logger.warning(f"模型 {model.model_type} 不支持获取特征重要性")
                return None
        except Exception as e:
            self.logger.error(f"获取特征重要性失败: {str(e)}")
            return None