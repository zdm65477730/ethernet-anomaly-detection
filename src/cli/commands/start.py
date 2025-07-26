import os
import typer
import time
import signal
from typing import Optional
from src.cli.utils import (
    print_success,
    print_error,
    print_warning,
    print_info,
    confirm,
    get_available_interfaces
)
from src.system.system_manager import SystemManager
from src.config.config_manager import ConfigManager
from src.utils.logger import init_logger

def is_process_running(pid_file: str) -> bool:
    """检查系统是否已在运行"""
    if not os.path.exists(pid_file):
        return False
        
    try:
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())
        
        # 尝试发送0信号检查进程是否存在
        os.kill(pid, 0)
        return True
    except (ValueError, OSError):
        # 无效PID或进程不存在
        os.remove(pid_file)
        return False

def write_pid_file(pid_file: str) -> None:
    """写入PID文件"""
    try:
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))
    except Exception as e:
        print_warning(f"无法写入PID文件 {pid_file}: {str(e)}")

def main(
    interface: Optional[str] = typer.Option(
        None, "--interface", "-i",
        help="网络接口名称"
    ),
    filter: str = typer.Option(
        "", "--filter", "-f",
        help="BPF过滤规则"
    ),
    config_dir: str = typer.Option(
        "config", "--config-dir", "-c",
        help="配置文件目录"
    ),
    background: bool = typer.Option(
        False, "--background", "-b",
        help="后台运行模式"
    ),
    log_level: Optional[str] = typer.Option(
        None, "--log-level", "-l",
        help="日志级别 (DEBUG, INFO, WARNING, ERROR)"
    )
):
    """启动异常检测系统"""
    # 检查配置目录
    if not os.path.isdir(config_dir):
        print_error(f"配置目录 {config_dir} 不存在，请先运行 'init' 命令初始化系统")
        raise typer.Exit(code=1)
    
    # 加载配置
    config = ConfigManager(config_dir=config_dir)
    
    # 确定PID文件路径
    pid_file = config.get("system.pid_file", "anomaly_detector.pid")
    
    # 检查是否已在运行
    if is_process_running(pid_file):
        print_error(f"系统似乎已在运行中 (PID文件: {pid_file})")
        print_info("如果确定系统未运行，请删除PID文件后重试")
        raise typer.Exit(code=1)
    
    # 检查网络接口
    if interface is None:
        # 从配置获取默认接口
        interface = config.get("capture.interface", "")
    
    if not interface:
        interfaces = get_available_interfaces()
        if not interfaces:
            print_error("未找到可用的网络接口")
            raise typer.Exit(code=1)
            
        print_info("未指定网络接口，可用接口:")
        for i, iface in enumerate(interfaces, 1):
            print(f"  {i}. {iface}")
        
        try:
            choice = input("请选择监听接口(输入序号): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(interfaces):
                interface = interfaces[idx]
            else:
                print_error("无效的选择")
                raise typer.Exit(code=1)
        except ValueError:
            print_error("无效的输入")
            raise typer.Exit(code=1)
    
    # 处理后台运行
    if background:
        try:
            # 简单的后台运行实现
            pid = os.fork()
            if pid > 0:
                # 父进程
                write_pid_file(pid_file)
                print_success(f"系统已在后台启动 (PID: {pid})")
                print_info(f"PID文件: {pid_file}")
                raise typer.Exit(code=0)
        except OSError as e:
            print_error(f"无法在后台运行: {str(e)}")
            raise typer.Exit(code=1)
    
    # 配置日志
    log_level = log_level or config.get("system.log_level", "INFO")
    init_logger(level=10 if log_level == "DEBUG" else 20)  # 10=DEBUG, 20=INFO
    
    # 写入PID文件
    write_pid_file(pid_file)
    
    # 创建并启动系统管理器
    system_manager = SystemManager()
    
    # 注册信号处理
    def handle_signal(signum, frame):
        print_info(f"接收到信号 {signum}，正在停止系统...")
        system_manager.stop()
        if os.path.exists(pid_file):
            os.remove(pid_file)
        print_success("系统已停止")
        raise typer.Exit(code=0)
    
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    system_started = False
    try:
        print_info(f"正在启动异常检测系统，监听接口: {interface}")
        if filter:
            print_info(f"使用BPF过滤规则: {filter}")
            
        system_started = system_manager.start(interface=interface, bpf_filter=filter)
        if system_started:
            print_success("系统启动成功")
        else:
            print_error("系统启动失败")
            raise typer.Exit(code=1)
        
        # 运行中
        try:
            while True:
                time.sleep(3600)  # 等待中断信号
        except KeyboardInterrupt:
            print_info("\n用户中断，正在停止系统...")
            system_manager.stop()
            
    except Exception as e:
        if not system_started:
            print_error(f"系统启动失败: {str(e)}")
        system_manager.stop()
        raise typer.Exit(code=1)
    finally:
        if os.path.exists(pid_file):
            os.remove(pid_file)

if __name__ == "__main__":
    typer.run(main)
