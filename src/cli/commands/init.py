import os
import typer
import yaml
import socket
import fcntl
import struct
from typing import List, Tuple
from src.data.data_generator import DataGenerator
from src.cli.utils import print_success, print_error, print_info, print_warning, confirm
from src.config.config_manager import ConfigManager

def get_available_interfaces() -> List[str]:
    """
    获取系统可用的网络接口列表
    
    Returns:
        List[str]: 可用网络接口名称列表
    """
    interfaces = []
    
    # 获取网络接口信息的常量
    SIOCGIFCONF = 0x8912  # 获取接口配置
    SIZE_OF_IFREQ = 40    # ifreq结构体大小
    
    # 创建socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    try:
        # 获取接口数量
        buf_size = 20 * SIZE_OF_IFREQ  # 20个接口的缓冲区
        ifconf = fcntl.ioctl(sock.fileno(), SIOCGIFCONF, bytes(buf_size))
        
        # 解析返回结果
        _, interfaces_bytes = ifconf[:4], ifconf[4:]
        
        # 提取接口名称
        for i in range(0, len(interfaces_bytes), SIZE_OF_IFREQ):
            interface_name = interfaces_bytes[i:i+16].decode('utf-8').strip('\x00')
            if interface_name:  # 忽略空名称
                interfaces.append(interface_name)
                
    except Exception as e:
        print_error(f"获取网络接口时出错: {str(e)}")
    finally:
        sock.close()
    
    return interfaces

def create_directories(dirs: List[str], overwrite: bool = False) -> Tuple[bool, List[str]]:
    """
    创建所需的目录结构
    
    Args:
        dirs: 要创建的目录列表
        overwrite: 如果为True，则覆盖已存在的目录
        
    Returns:
        Tuple[bool, List[str]]: (是否成功, 失败的目录列表)
    """
    failed_dirs = []
    
    for dir_path in dirs:
        try:
            # 如果目录已存在且不强制覆盖，则跳过
            if os.path.exists(dir_path) and not overwrite:
                continue
                
            # 创建目录及其父目录
            os.makedirs(dir_path, exist_ok=True)
            
        except Exception as e:
            print_error(f"无法创建目录 {dir_path}: {str(e)}")
            failed_dirs.append(dir_path)
    
    # 返回成功状态和失败列表
    return len(failed_dirs) == 0, failed_dirs

app = typer.Typer(help="系统初始化命令", invoke_without_command=True)

# 没有需要修改的内容

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

@app.callback()
def main(
    ctx: typer.Context,
    config_dir: str = typer.Option(
        "config", "--config-dir", "-c",
        help="配置文件目录"
    ),
    data_dir: str = typer.Option(
        ".", "--data-dir", "-d",
        help="数据目录"
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="强制覆盖已存在的配置文件"
    ),
    generate_data: bool = typer.Option(
        False, "--generate-data", "-g",
        help="生成示例数据"
    ),
    samples: int = typer.Option(
        1000, "--samples", "-s",
        help="生成的样本数量"
    )
):
    """
    初始化系统配置和目录结构
    """
    # 如果是直接调用init命令而不是子命令，则执行初始化
    if ctx.invoked_subcommand is None:
        print_info("正在初始化异常检测系统...")
        
        try:
            # 创建必要的目录
            dirs_to_create = [
                config_dir,
                "data/raw",
                "data/processed",
                "data/test",
                "models",
                "logs",
                "reports/evaluations",
                "reports/detections"
            ]
            
            success, failed = create_directories(dirs_to_create, overwrite=force)
            if not success:
                print_error(f"创建目录失败: {', '.join(failed)}")
                raise typer.Exit(code=1)
            
            # 创建默认配置文件
            config_file = f"{config_dir}/config.yaml"
            if not force and ConfigManager.config_file_exists(config_dir):
                if not confirm("配置文件已存在，是否覆盖?"):
                    print_info("跳过配置文件创建")
                else:
                    ConfigManager.create_default_config(config_dir)
                    print_success(f"已创建配置文件: {config_file}")
            else:
                ConfigManager.create_default_config(config_dir)
                print_success(f"已创建配置文件: {config_file}")
            
            # 获取可用网络接口
            interfaces = get_available_interfaces()
            if interfaces:
                print_info("可用网络接口:")
                for interface in interfaces:
                    print(f"  - {interface}")
            else:
                print_warning("未找到可用网络接口")
            
            # 生成示例数据（如果需要）
            if generate_data:
                print_info(f"正在生成 {samples} 个示例数据样本...")
                try:
                    generator = DataGenerator()
                    result = generator.generate(
                        num_samples=samples,
                        output_dir="data/processed"
                    )
                    print_success("示例数据生成完成!")
                    
                    # 同时将测试数据复制到data/test目录以支持模型评估
                    import shutil
                    import pandas as pd
                    if "test_path" in result:
                        # 如果生成时已经分割了训练集和测试集
                        shutil.copy2(result["test_path"], "data/test/test_data.csv")
                        print_success("测试数据已复制到 data/test 目录!")
                    else:
                        # 如果没有分割，则从完整数据集中复制一部分作为测试数据
                        full_data_path = result.get("full_path", "data/processed/full_data.csv")
                        df = pd.read_csv(full_data_path)
                        # 取20%作为测试数据
                        test_df = df.sample(frac=0.2, random_state=42)
                        test_df.to_csv("data/test/test_data.csv", index=False)
                        print_success("测试数据已生成并保存到 data/test 目录!")
                        
                except Exception as e:
                    print_error(f"生成示例数据时出错: {str(e)}")
            
        except Exception as e:
            print_error(f"初始化系统时发生错误: {str(e)}")
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
    app()
