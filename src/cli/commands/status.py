import os
import typer
import json
from typing import Optional
from src.cli.utils import (
    print_success,
    print_error,
    print_info,
    print_warning
)
from src.system.system_manager import SystemManager
from src.config.config_manager import ConfigManager

def format_status(status: dict, indent: int = 0) -> str:
    """格式化状态信息为可读字符串"""
    result = []
    indent_str = "  " * indent
    
    for key, value in status.items():
        if isinstance(value, dict):
            result.append(f"{indent_str}{key}:")
            result.append(format_status(value, indent + 1))
        else:
            result.append(f"{indent_str}{key}: {value}")
    
    return "\n".join(result)

def main(
    pid_file: Optional[str] = typer.Option(
        None, "--pid-file", "-p",
        help="PID文件路径"
    ),
    config_dir: str = typer.Option(
        "config", "--config-dir", "-c",
        help="配置文件目录"
    ),
    format: str = typer.Option(
        "text", "--format", "-f",
        help="输出格式 (text, json)"
    )
):
    """查询异常检测系统运行状态"""
    # 确定PID文件路径
    if pid_file is None:
        try:
            config = ConfigManager(config_dir=config_dir)
            pid_file = config.get("system.pid_file", "anomaly_detector.pid")
        except Exception:
            pid_file = "anomaly_detector.pid"
    
    # 检查系统是否在运行
    running = False
    pid = None
    
    if os.path.exists(pid_file):
        try:
            with open(pid_file, "r") as f:
                pid = int(f.read().strip())
            
            # 检查进程是否存在
            os.kill(pid, 0)
            running = True
        except (ValueError, OSError):
            # PID文件无效或进程不存在
            pass
    
    if not running:
        if format == "json":
            print(json.dumps({"status": "stopped"}, indent=2))
        else:
            print_warning("异常检测系统当前未运行")
        raise typer.Exit(code=0 if not running else 1)
    
    # 如果系统运行中，获取详细状态
    try:
        system_manager = SystemManager()
        status = system_manager.get_status()
        
        # 添加基本信息
        status["system"] = {
            "status": "running",
            "pid": pid
        }
        
        if format == "json":
            print(json.dumps(status, indent=2))
        else:
            print_success("异常检测系统正在运行中")
            print(format_status(status))
            
    except Exception as e:
        if format == "json":
            print(json.dumps({
                "status": "running",
                "pid": pid,
                "error": f"无法获取详细状态: {str(e)}"
            }, indent=2))
        else:
            print_success(f"异常检测系统正在运行中 (PID: {pid})")
            print_warning(f"无法获取详细状态: {str(e)}")

if __name__ == "__main__":
    typer.run(main)
