import os
import typer
import yaml
from typing import Optional
from src.cli.utils import (
    print_success,
    print_error,
    print_warning,
    print_info,
    confirm,
    create_directories,
    get_available_interfaces,
    validate_file_path
)
from src.config.config_manager import ConfigManager

# 初始化所需的目录结构
REQUIRED_DIRECTORIES = [
    "config",
    "data/raw",
    "data/processed",
    "data/test",
    "logs",
    "models/xgboost",
    "models/random_forest",
    "models/lstm",
    "reports/evaluation/figures",
    "reports/anomalies"
]

# 默认配置模板
DEFAULT_CONFIG = {
    "system": {
        "log_level": "INFO",
        "pid_file": "anomaly_detector.pid",
        "max_restarts": 3
    },
    "capture": {
        "interface": "",  # 将由用户选择或留空
        "bpf_filter": "",
        "buffer_size": 1024 * 1024,  # 1MB
        "snaplen": 65535,  # 捕获完整数据包
        "promiscuous": False
    },
    "session": {
        "timeout": 300,  # 会话超时时间(秒)
        "max_packets_per_session": 1000,  # 每个会话最多保存的数据包数
        "cleanup_interval": 60  # 会话清理间隔(秒)
    },
    "features": {
        "enabled_stat_features": [
            "packet_size", "protocol", "payload_size",
            "tcp_flags", "udp_length", "icmp_type"
        ],
        "temporal_window_size": 60,  # 时序特征窗口大小(秒)
        "temporal_window_step": 10   # 时序特征窗口步长(秒)
    },
    "detection": {
        "threshold": 0.8,  # 异常判断阈值
        "alert_email": "",  # 告警邮件地址
        "smtp_server": "smtp.example.com",
        "smtp_port": 587,
        "smtp_username": "",
        "smtp_password": ""
    },
    "training": {
        "check_interval": 3600,  # 持续训练检查间隔(秒)
        "min_samples": 1000,  # 触发训练的最小样本数
        "test_size": 0.2,  # 测试集比例
        "cross_validation_folds": 5,
        "retrain_threshold": 0.05  # 模型性能提升阈值(5%)
    },
    "model": {
        "type": "xgboost",  # 默认模型类型
        "save_interval": 3600,  # 模型保存间隔(秒)
        "max_versions": 10  # 保留的模型版本数
    }
}

# 模型配置模板
MODEL_CONFIG = {
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss"
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "n_jobs": -1
    },
    "lstm": {
        "input_dim": 10,
        "hidden_dim": 32,
        "output_dim": 1,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10
    }
}

# 检测规则配置模板
DETECTION_RULES = {
    "size_based": {
        "enabled": True,
        "max_normal_size": 1500  # 超过此大小的包视为异常
    },
    "protocol_based": {
        "enabled": True,
        "suspicious_protocols": [1]  # ICMP默认视为可疑协议
    },
    "rate_based": {
        "enabled": True,
        "max_packets_per_second": 100  # 每秒超过此数量视为异常
    }
}

def main(
    force: bool = typer.Option(
        False, "--force", "-f",
        help="强制覆盖现有配置和目录"
    ),
    config_dir: str = typer.Option(
        "config", "--config-dir", "-c",
        help="配置文件目录"
    ),
    data_dir: str = typer.Option(
        "data", "--data-dir", "-d",
        help="数据存储目录"
    )
):
    """初始化系统配置和目录结构"""
    print_info("开始初始化异常检测系统...")
    
    # 确认是否覆盖现有文件
    if not force:
        existing_config = os.path.exists(os.path.join(config_dir, "config.yaml"))
        existing_data = os.path.exists(data_dir)
        
        if existing_config or existing_data:
            if not confirm(
                f"检测到现有配置或数据目录，{'将被覆盖' if force else '是否继续?'}"
            ):
                print_info("初始化已取消")
                raise typer.Exit(code=0)
    
    # 创建所需目录
    dirs_to_create = [
        os.path.join(config_dir),
        os.path.join(data_dir, "raw"),
        os.path.join(data_dir, "processed"),
        os.path.join(data_dir, "test"),
        "logs",
        "models/xgboost",
        "models/random_forest",
        "models/lstm",
        "reports/evaluation/figures",
        "reports/anomalies"
    ]
    
    success, failed = create_directories(dirs_to_create, overwrite=force)
    if not success:
        print_error(f"初始化失败，以下目录创建失败: {', '.join(failed)}")
        raise typer.Exit(code=1)
    
    # 获取可用网络接口
    interfaces = get_available_interfaces()
    selected_interface = ""
    
    if interfaces:
        print_info("可用网络接口:")
        for i, iface in enumerate(interfaces, 1):
            print(f"  {i}. {iface}")
        
        try:
            choice = input("请选择默认监听接口(输入序号，直接回车则留空): ").strip()
            if choice:
                idx = int(choice) - 1
                if 0 <= idx < len(interfaces):
                    selected_interface = interfaces[idx]
                    print_info(f"已选择默认接口: {selected_interface}")
        except (ValueError, IndexError):
            print_warning("无效选择，将使用空接口配置")
    
    # 生成配置文件
    config = DEFAULT_CONFIG.copy()
    config["capture"]["interface"] = selected_interface
    
    # 保存主配置
    main_config_path = os.path.join(config_dir, "config.yaml")
    try:
        with open(main_config_path, "w") as f:
            yaml.dump(config, f, sort_keys=False, default_flow_style=False)
        print_success(f"已生成主配置文件: {main_config_path}")
    except Exception as e:
        print_error(f"生成主配置文件失败: {str(e)}")
        raise typer.Exit(code=1)
    
    # 保存模型配置
    model_config_path = os.path.join(config_dir, "model_config.yaml")
    try:
        with open(model_config_path, "w") as f:
            yaml.dump(MODEL_CONFIG, f, sort_keys=False, default_flow_style=False)
        print_success(f"已生成模型配置文件: {model_config_path}")
    except Exception as e:
        print_error(f"生成模型配置文件失败: {str(e)}")
        raise typer.Exit(code=1)
    
    # 保存检测规则配置
    rules_config_path = os.path.join(config_dir, "detection_rules.yaml")
    try:
        with open(rules_config_path, "w") as f:
            yaml.dump(DETECTION_RULES, f, sort_keys=False, default_flow_style=False)
        print_success(f"已生成检测规则配置文件: {rules_config_path}")
    except Exception as e:
        print_error(f"生成检测规则配置文件失败: {str(e)}")
        raise typer.Exit(code=1)
    
    print_success("系统初始化完成！")
    print_info("下一步可以使用 'anomaly-detector start' 命令启动系统")

if __name__ == "__main__":
    typer.run(main)
