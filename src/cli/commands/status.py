import os
import typer
import json
import psutil
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

def _check_process_status(pid_file: str, process_name: str) -> bool:
    """检查指定PID文件的进程状态"""
    if not os.path.exists(pid_file):
        print_info(f"{process_name}未运行 (PID文件不存在)")
        return False
        
    # 读取PID
    try:
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())
    except (ValueError, IOError) as e:
        print_warning(f"读取{process_name} PID文件失败: {e}")
        return False
        
    # 检查进程是否存在
    try:
        # 使用psutil检查进程，它可以更好地处理权限问题
        process = psutil.Process(pid)
        if process.is_running():
            print_success(f"{process_name}正在运行 (PID: {pid})")
            return True
        else:
            print_info(f"{process_name}未运行 (进程 {pid} 不存在)")
            # 删除无效的PID文件
            try:
                os.remove(pid_file)
            except:
                pass
            return False
    except psutil.NoSuchProcess:
        print_info(f"{process_name}未运行 (进程 {pid} 不存在)")
        # 删除无效的PID文件
        try:
            os.remove(pid_file)
        except:
            pass
        return False
    except psutil.AccessDenied:
        # 权限不足，但进程存在
        print_success(f"{process_name}正在运行 (PID: {pid})，但无权限访问进程详情")
        return True
    except Exception as e:
        print_warning(f"检查{process_name}进程状态时出错: {e}")
        return False

status_app = typer.Typer(help="查看系统状态")

def main(
    pid_file: Optional[str] = typer.Option(
        None, "--pid-file", "-p",
        help="PID文件路径"
    ),
    config_dir: str = typer.Option(
        "config", "--config-dir", "-c",
        help="配置文件目录"
    ),
    detail: bool = typer.Option(
        False, "--detail", "-d",
        help="显示详细信息"
    )
):
    """
    查看系统运行状态
    """
    try:
        # 检查主系统进程
        main_pid_file = pid_file or "anomaly_detector.pid"
        main_running = _check_process_status(main_pid_file, "主系统")
        
        # 检查持续训练进程
        continuous_pid_file = "continuous_training.pid"
        continuous_running = _check_process_status(continuous_pid_file, "持续训练")
        
        # 如果没有任何进程在运行，直接返回
        if not main_running and not continuous_running:
            return
            
        # 加载配置
        try:
            config = ConfigManager(config_dir=config_dir)
        except Exception as e:
            print_warning(f"加载配置失败: {e}")
            config = None
            
        # 显示详细信息
        if detail and config:
            try:
                system_manager = SystemManager(config=config)
                status = system_manager.get_system_status()
                print_info("系统详细状态:")
                print(format_status(status))
            except Exception as e:
                print_warning(f"获取系统详细状态失败: {e}")
                
    except Exception as e:
        print_error(f"检查系统状态时发生错误: {str(e)}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    status_app()