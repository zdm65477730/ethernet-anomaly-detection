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

class ModelFactory:
    """模型工厂，统一管理模型的创建、训练、保存和加载"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
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
        
        # 确保模型目录存在
        os.makedirs(self.models_dir, exist_ok=True)

    def get_supported_models(self) -> List[str]:
        """返回支持的模型类型列表"""
        return list(self._model_classes.keys())

    def create_model(self, model_type: str,** kwargs) -> BaseModel:
        """
        创建新模型实例
        
        参数:
            model_type: 模型类型
            **kwargs: 模型参数
            
        返回:
            模型实例
        """
        model_type = model_type.lower()
        if model_type not in self._model_classes:
            raise ValueError(
                f"不支持的模型类型: {model_type}，支持的类型: {self.get_supported_models()}"
            )
        
        try:
            model = self._model_classes[model_type](** kwargs)
            self.logger.info(f"已创建 {model_type} 模型实例")
            return model
        except Exception as e:
            self.logger.error(f"创建 {model_type} 模型失败: {str(e)}", exc_info=True)
            raise

    def save_model(
        self,
        model: BaseModel,
        protocol: Optional[str] = None,
        create_latest_link: bool = True
    ) -> str:
        """
        保存模型到文件系统
        
        参数:
            model: 模型实例
            protocol: 协议名称（用于分类存储）
            create_latest_link: 是否创建指向最新模型的软链接
            
        返回:
            模型保存路径
        """
        if not isinstance(model, BaseModel):
            raise ValueError("model必须是BaseModel的子类实例")
            
        if not model.is_trained:
            self.logger.warning("保存未训练的模型")
        
        # 构建保存路径
        timestamp = int(time.time())
        model_type = model.model_type
        
        # 按协议和模型类型组织目录
        if protocol:
            model_dir = os.path.join(self.models_dir, protocol, model_type)
        else:
            model_dir = os.path.join(self.models_dir, model_type)
        
        os.makedirs(model_dir, exist_ok=True)
        
        # 模型文件名（包含时间戳）
        model_filename = f"model_{timestamp}.pkl" if model_type != "lstm" and model_type != "mlp" else f"model_{timestamp}"
        model_path = os.path.join(model_dir, model_filename)
        
        # 保存模型
        try:
            model.save(model_path)
            
            # 创建指向最新模型的软链接
            if create_latest_link:
                latest_link = os.path.join(model_dir, "latest")
                # 移除已存在的链接
                if os.path.exists(latest_link) or os.path.islink(latest_link):
                    os.unlink(latest_link)
                # 创建新链接
                os.symlink(model_path, latest_link)
                self.logger.debug(f"已创建最新模型软链接: {latest_link} -> {model_path}")
                
            self.logger.info(f"模型已保存到 {model_path}")
            return model_path
        except Exception as e:
            self.logger.error(f"保存模型失败: {str(e)}", exc_info=True)
            raise

    def load_model(
        self,
        model_type: str,
        protocol: Optional[str] = None,
        version: Optional[str] = "latest"
    ) -> BaseModel:
        """
        从文件系统加载模型
        
        参数:
            model_type: 模型类型
            protocol: 协议名称（用于定位模型）
            version: 模型版本，"latest"表示最新版本
            
        返回:
            加载的模型实例
        """
        model_type = model_type.lower()
        if model_type not in self._model_classes:
            raise ValueError(
                f"不支持的模型类型: {model_type}，支持的类型: {self.get_supported_models()}"
            )
        
        # 构建模型路径
        if protocol:
            model_dir = os.path.join(self.models_dir, protocol, model_type)
        else:
            model_dir = os.path.join(self.models_dir, model_type)
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"模型目录不存在: {model_dir}")
        
        # 确定模型文件路径
        if version == "latest":
            model_path = os.path.join(model_dir, "latest")
            # 检查软链接是否存在
            if not os.path.exists(model_path) or not os.path.islink(model_path):
                # 尝试查找最新的模型文件
                model_files = self._find_model_files(model_dir, model_type)
                if not model_files:
                    raise FileNotFoundError(f"在 {model_dir} 中未找到 {model_type} 模型文件")
                model_path = model_files[-1]  # 最新的文件
        else:
            model_path = os.path.join(model_dir, version)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载模型
        try:
            model_class = self._model_classes[model_type]
            model = model_class.load(model_path)
            self.logger.info(f"已从 {model_path} 加载 {model_type} 模型")
            return model
        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}", exc_info=True)
            raise

    def get_model_versions(
        self,
        model_type: str,
        protocol: Optional[str] = None
    ) -> List[str]:
        """获取指定模型类型和协议的所有版本"""
        model_type = model_type.lower()
        
        # 构建模型目录
        if protocol:
            model_dir = os.path.join(self.models_dir, protocol, model_type)
        else:
            model_dir = os.path.join(self.models_dir, model_type)
        
        if not os.path.exists(model_dir):
            return []
            
        # 查找所有模型文件
        model_files = self._find_model_files(model_dir, model_type)
        return [os.path.basename(f) for f in model_files]

    def _find_model_files(self, model_dir: str, model_type: str) -> List[str]:
        """查找目录中的所有模型文件并按时间戳排序"""
        # 模型文件的时间戳正则表达式
        pattern = re.compile(r"model_(\d+)\.(pkl|h5)?")
        
        model_files = []
        for filename in os.listdir(model_dir):
            match = pattern.match(filename)
            if match:
                file_path = os.path.join(model_dir, filename)
                # 检查是否是文件
                if os.path.isfile(file_path):
                    model_files.append((int(match.group(1)), file_path))
        
        # 按时间戳排序（升序）
        model_files.sort(key=lambda x: x[0])
        # 返回文件路径列表（最新的在最后）
        return [fp for (ts, fp) in model_files]

    def delete_model_version(
        self,
        model_type: str,
        version: str,
        protocol: Optional[str] = None
    ) -> bool:
        """删除指定版本的模型"""
        model_type = model_type.lower()
        
        # 构建模型路径
        if protocol:
            model_dir = os.path.join(self.models_dir, protocol, model_type)
        else:
            model_dir = os.path.join(self.models_dir, model_type)
            
        model_path = os.path.join(model_dir, version)
        
        if not os.path.exists(model_path):
            self.logger.warning(f"模型文件不存在: {model_path}")
            return False
            
        try:
            # 如果是Keras模型，需要删除相关文件
            if model_type in ["lstm", "mlp"]:
                if os.path.exists(f"{model_path}.meta"):
                    os.remove(f"{model_path}.meta")
                # 删除整个模型目录（Keras模型保存为目录）
                if os.path.isdir(model_path):
                    import shutil
                    shutil.rmtree(model_path)
            else:
                # 普通模型文件
                os.remove(model_path)
                
            self.logger.info(f"已删除模型: {model_path}")
            
            # 如果删除的是最新版本，更新软链接
            latest_link = os.path.join(model_dir, "latest")
            if os.path.islink(latest_link) and os.readlink(latest_link) == model_path:
                os.unlink(latest_link)
                # 查找剩余的最新模型
                remaining_models = self._find_model_files(model_dir, model_type)
                if remaining_models:
                    os.symlink(remaining_models[-1], latest_link)
                    self.logger.info(f"已更新最新模型软链接到: {remaining_models[-1]}")
            
            return True
        except Exception as e:
            self.logger.error(f"删除模型失败: {str(e)}", exc_info=True)
            return False

    def get_model_metadata(
        self,
        model_type: str,
        protocol: Optional[str] = None,
        version: str = "latest"
    ) -> Dict[str, any]:
        """获取模型元数据"""
        model = self.load_model(model_type, protocol, version)
        return model.get_metadata()
