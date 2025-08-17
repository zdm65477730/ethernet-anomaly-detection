#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异常检测系统CLI主程序
"""
import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

import typer
from typing import Optional
from src.config.config_manager import ConfigManager
from src.utils.logger import get_logger
from src.cli.commands import train, generate_test_data, pcap_to_csv, report, start

# 初始化日志
logger = get_logger(__name__)

# 创建Typer应用
app = typer.Typer(no_args_is_help=True, add_completion=False)

# 添加子命令
train_app = train.train_app
generate_test_data_app = generate_test_data.app
pcap_to_csv_app = pcap_to_csv.app
report_app = report.app
start_app = start.app

app.add_typer(train_app, name="train", help="训练模型")
app.add_typer(generate_test_data_app, name="generate-test-data", help="生成测试数据")
app.add_typer(pcap_to_csv_app, name="pcap-to-csv", help="将PCAP文件转换为CSV格式")
app.add_typer(report_app, name="report", help="生成检测报告")
app.add_typer(start_app, name="start", help="启动异常检测系统")

@app.command()
def init():
    """初始化系统配置"""
    typer.echo("正在初始化系统配置...")
    
    # 创建必要的目录
    dirs_to_create = ["config", "data", "models", "logs", "reports"]
    for dir_name in dirs_to_create:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            typer.echo(f"已创建目录: {dir_name}")
    
    # 检查配置文件是否存在
    config_files = [
        "config/config.yaml",
        "config/model_config.yaml", 
        "config/detection_rules.yaml",
        "config/self_driving_config.yaml"
    ]
    
    missing_configs = []
    for config_file in config_files:
        if not os.path.exists(config_file):
            missing_configs.append(config_file)
    
    if missing_configs:
        typer.echo("以下配置文件缺失:")
        for config_file in missing_configs:
            typer.echo(f"  - {config_file}")
        typer.echo("请运行 'anomaly-detector generate-default-config' 命令生成默认配置文件")
    else:
        typer.echo("系统配置已存在")
    
    typer.echo("初始化完成!")

@app.callback()
def main(
    ctx: typer.Context,
    config_dir: str = typer.Option("config", "--config-dir", "-c", help="配置文件目录"),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="日志级别 (DEBUG, INFO, WARNING, ERROR)")
):
    """异常检测系统CLI工具"""
    # 设置日志级别
    import logging
    logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # 加载配置
    try:
        config = ConfigManager(config_dir=config_dir)
        ctx.ensure_object(dict)
        ctx.obj["config"] = config
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()