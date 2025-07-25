import os
import yaml
from typing import Dict, Any, Optional
from src.utils.logger import get_logger

class ConfigManager:
    """配置管理器，负责加载、保存和访问配置文件"""
    
    def __init__(self, config_dir: str = "config"):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = os.path.abspath(config_dir)
        self.logger = get_logger("config_manager")
        
        # 配置文件路径
        self.main_config_path = os.path.join(self.config_dir, "config.yaml")
        self.model_config_path = os.path.join(self.config_dir, "model_config.yaml")
        self.detection_rules_path = os.path.join(self.config_dir, "detection_rules.yaml")
        
        # 配置数据
        self._config: Dict[str, Any] = {}
        self._model_config: Dict[str, Any] = {}
        self._detection_rules: Dict[str, Any] = {}
        
        # 加载配置
        self.load()
    
    def load(self) -> None:
        """加载所有配置文件"""
        # 确保配置目录存在
        if not os.path.exists(self.config_dir):
            self.logger.warning(f"配置目录 {self.config_dir} 不存在，将使用默认配置")
            self._init_default_configs()
            return
        
        # 加载主配置
        try:
            if os.path.exists(self.main_config_path):
                with open(self.main_config_path, "r") as f:
                    self._config = yaml.safe_load(f) or {}
                self.logger.info(f"已加载主配置: {self.main_config_path}")
            else:
                self.logger.warning(f"主配置文件 {self.main_config_path} 不存在，使用默认配置")
                self._init_default_main_config()
        except Exception as e:
            self.logger.error(f"加载主配置失败: {str(e)}，使用默认配置")
            self._init_default_main_config()
        
        # 加载模型配置
        try:
            if os.path.exists(self.model_config_path):
                with open(self.model_config_path, "r") as f:
                    self._model_config = yaml.safe_load(f) or {}
                self.logger.info(f"已加载模型配置: {self.model_config_path}")
            else:
                self.logger.warning(f"模型配置文件 {self.model_config_path} 不存在，使用默认配置")
                self._init_default_model_config()
        except Exception as e:
            self.logger.error(f"加载模型配置失败: {str(e)}，使用默认配置")
            self._init_default_model_config()
        
        # 加载检测规则
        try:
            if os.path.exists(self.detection_rules_path):
                with open(self.detection_rules_path, "r") as f:
                    self._detection_rules = yaml.safe_load(f) or {}
                self.logger.info(f"已加载检测规则: {self.detection_rules_path}")
            else:
                self.logger.warning(f"检测规则文件 {self.detection_rules_path} 不存在，使用默认配置")
                self._init_default_detection_rules()
        except Exception as e:
            self.logger.error(f"加载检测规则失败: {str(e)}，使用默认配置")
            self._init_default_detection_rules()
    
    def _init_default_main_config(self) -> None:
        """初始化默认主配置"""
        self._config = {
            "general": {
                "log_level": "INFO",
                "log_dir": "logs",
                "report_dir": "reports"
            },
            "network": {
                "interface": "eth0",
                "bpf_filter": "",
                "packet_buffer_size": 10000
            },
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "test_dir": "data/test",
                "storage_format": "parquet",  # parquet 或 csv
                "retention_days": 30
            },
            "features": {
                "enabled_stat_features": [
                    "packet_size", "protocol", "payload_size",
                    "tcp_flags", "window_size", "packet_size_std"
                ],
                "temporal_window_size": 60,  # 秒
                "temporal_window_step": 10   # 秒
            },
            "detection": {
                "threshold": 0.8,
                "alert_email": "",
                "alert_smtp_server": "smtp.example.com",
                "alert_smtp_port": 587,
                "alert_smtp_username": "",
                "alert_smtp_password": ""
            },
            "training": {
                "model_dir": "models",
                "check_interval": 3600,  # 检查新数据的间隔(秒)
                "min_samples": 1000,     # 触发训练的最小样本数
                "cross_validation_folds": 5,
                "test_size": 0.2,
                "class_weight": None
            }
        }
    
    def _init_default_model_config(self) -> None:
        """初始化默认模型配置"""
        self._model_config = {
            "xgboost": {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "random_state": 42,
                "use_label_encoder": False,
                "eval_metric": "logloss"
            },
            "random_forest": {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "random_state": 42,
                "n_jobs": -1
            },
            "lstm": {
                "input_dim": 10,
                "hidden_dim": 64,
                "output_dim": 1,
                "layers": 2,
                "dropout": 0.2,
                "batch_size": 32,
                "epochs": 10,
                "learning_rate": 0.001
            },
            "logistic_regression": {
                "C": 1.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 100,
                "random_state": 42
            }
        }
    
    def _init_default_detection_rules(self) -> None:
        """初始化默认检测规则"""
        self._detection_rules = {
            "size_based": {
                "enabled": True,
                "max_normal_packet_size": 1500  # 字节
            },
            "protocol_based": {
                "enabled": True,
                "suspicious_protocols": [1]  # ICMP协议
            },
            "rate_based": {
                "enabled": True,
                "max_packets_per_second": 100,
                "window_seconds": 10
            },
            "tcp_flags": {
                "enabled": True,
                "suspicious_combinations": [
                    "FIN,PSH,URG",  # nmap XMAS扫描
                    "FIN"           # nmap FIN扫描
                ]
            }
        }
    
    def _init_default_configs(self) -> None:
        """初始化所有默认配置"""
        self._init_default_main_config()
        self._init_default_model_config()
        self._init_default_detection_rules()
    
    def save(self) -> None:
        """保存所有配置文件"""
        # 确保配置目录存在
        os.makedirs(self.config_dir, exist_ok=True)
        
        # 保存主配置
        try:
            with open(self.main_config_path, "w") as f:
                yaml.dump(self._config, f, sort_keys=False, indent=2)
            self.logger.info(f"已保存主配置: {self.main_config_path}")
        except Exception as e:
            self.logger.error(f"保存主配置失败: {str(e)}")
        
        # 保存模型配置
        try:
            with open(self.model_config_path, "w") as f:
                yaml.dump(self._model_config, f, sort_keys=False, indent=2)
            self.logger.info(f"已保存模型配置: {self.model_config_path}")
        except Exception as e:
            self.logger.error(f"保存模型配置失败: {str(e)}")
        
        # 保存检测规则
        try:
            with open(self.detection_rules_path, "w") as f:
                yaml.dump(self._detection_rules, f, sort_keys=False, indent=2)
            self.logger.info(f"已保存检测规则: {self.detection_rules_path}")
        except Exception as e:
            self.logger.error(f"保存检测规则失败: {str(e)}")
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        通过点路径获取配置值
        
        Args:
            path: 配置路径，如 "network.interface"
            default: 默认值
        
        Returns:
            配置值或默认值
        """
        # 确定要查询的配置字典
        if path.startswith("model."):
            parts = path.split(".")[1:]
            config_dict = self._model_config
        elif path.startswith("rule."):
            parts = path.split(".")[1:]
            config_dict = self._detection_rules
        else:
            parts = path.split(".")
            config_dict = self._config
        
        # 遍历路径获取值
        current = config_dict
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current
    
    def set(self, path: str, value: Any) -> None:
        """
        通过点路径设置配置值
        
        Args:
            path: 配置路径，如 "network.interface"
            value: 要设置的值
        """
        # 确定要修改的配置字典
        if path.startswith("model."):
            parts = path.split(".")[1:]
            config_dict = self._model_config
        elif path.startswith("rule."):
            parts = path.split(".")[1:]
            config_dict = self._detection_rules
        else:
            parts = path.split(".")
            config_dict = self._config
        
        # 遍历路径设置值
        current = config_dict
        for i, part in enumerate(parts[:-1]):
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        
        # 设置最终值
        current[parts[-1]] = value
        self.logger.debug(f"设置配置 {path} = {value}")
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """获取指定模型的配置"""
        return self._model_config.get(model_type, {}).copy()
    
    def get_detection_rules(self) -> Dict[str, Any]:
        """获取所有检测规则"""
        return self._detection_rules.copy()
    
    def __str__(self) -> str:
        """返回配置的字符串表示"""
        return f"ConfigManager(config_dir={self.config_dir})"
    