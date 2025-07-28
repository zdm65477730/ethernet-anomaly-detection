import os
import sys

# 在绝对最早期设置环境变量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 在绝对最早期重定向stderr
class DevNull:
    def write(self, msg):
        # 检查是否包含TensorFlow相关关键词
        tf_keywords = [
            'tensorflow', 'cuda', 'cudnn', 'cufft', 'cublas', 
            'absl', 'computation_placer', 'oneDNN', 'stream_executor',
            'external/local_xla', 'eigen', 'registering factory',
            'attempting to register', 'please check linkage'
        ]
        
        msg_lower = str(msg).lower()
        # 如果包含任何TF关键词，直接丢弃
        for keyword in tf_keywords:
            if keyword in msg_lower:
                return
        # 否则正常输出到原始stderr
        sys.__stderr__.write(msg)
    
    def close(self):
        pass
    
    def flush(self):
        sys.__stderr__.flush()
    
    def fileno(self):
        return sys.__stderr__.fileno()

# 重定向stderr到我们的过滤器
sys.stderr = DevNull()

import typer
from typing import Optional
import signal
import logging
import time
from src.system.system_manager import SystemManager
from src.config.config_manager import ConfigManager
from src.cli.utils import print_info, print_success, print_error, print_warning
from src.utils.logger import setup_logging, get_logger

app = typer.Typer(help="系统控制命令", invoke_without_command=True)

# 全局变量用于信号处理
_running = True
_system_manager = None  # 添加全局系统管理器引用

def _signal_handler(signum, frame):
    """信号处理函数"""
    global _running
    global _system_manager
    print_info(f"收到信号 {signum}，正在停止系统...")
    
    # 设置运行标志为False
    _running = False
    
    # 停止系统管理器
    if _system_manager and _system_manager.is_running:
        try:
            _system_manager.stop()
        except Exception as e:
            print_error(f"停止系统管理器时出错: {e}")
    
    # 确保退出程序
    sys.exit(0)

def _create_pid_file():
    """创建PID文件"""
    pid_file = "anomaly_detector.pid"
    try:
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))
        return pid_file
    except Exception as e:
        print_warning(f"创建PID文件失败: {e}")
        return None

def _remove_pid_file():
    """删除PID文件"""
    pid_file = "anomaly_detector.pid"
    try:
        if os.path.exists(pid_file):
            os.remove(pid_file)
    except Exception as e:
        print_warning(f"删除PID文件失败: {e}")

@app.callback()
def main(
    ctx: typer.Context,
    config_dir: str = typer.Option(
        "config", "--config-dir", "-c",
        help="配置文件目录"
    ),
    daemon: bool = typer.Option(
        False, "--daemon", "-d",
        help="以守护进程模式运行"
    ),
    interface: Optional[str] = typer.Option(
        None, "--interface", "-i",
        help="网络接口名称"
    ),
    filter: Optional[str] = typer.Option(
        None, "--filter", "-f",
        help="BPF过滤规则"
    ),
    offline_file: Optional[str] = typer.Option(
        None, "--offline-file", "-o",
        help="离线pcap文件路径"
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", "-l",
        help="日志级别"
    ),
):
    """
    启动异常检测系统
    
    如果直接运行start命令而没有指定子命令，
    则会启动异常检测系统。
    
    示例:
    
    \b
    $ python -m src.cli.commands.start
    $ python -m src.cli.commands.start --interface eth0
    """
    # 如果没有子命令，直接启动系统
    if ctx.invoked_subcommand is None:
        start_system(
            config_dir=config_dir,
            daemon=daemon or False,
            interface=interface or None,
            filter=filter or None,
            offline_file=offline_file or None,
            log_level=log_level or "INFO"
        )

def start_system(
    config_dir: str = "config",
    interface: Optional[str] = None,
    filter: Optional[str] = None,
    offline_file: Optional[str] = None,
    log_level: str = "INFO",
    daemon: bool = False
):
    """启动异常检测系统"""
    global _running
    global _system_manager
    
    try:
        # 设置日志
        setup_logging(log_level.upper())
        logger = get_logger(__name__)
        
        # 加载配置
        config = ConfigManager(config_dir=config_dir)
        
        # 注册信号处理器（提前注册以确保能捕获信号）
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        # 创建系统管理器
        system_manager = SystemManager(config=config)
        _system_manager = system_manager  # 保存全局引用
        
        # 解除信号阻塞
        signal.pthread_sigmask(signal.SIG_UNBLOCK, [signal.SIGINT, signal.SIGTERM])
        
        # 启动系统
        print_info("正在启动异常检测系统...")
        success = system_manager.start(
            interface=interface or config.get("network.interface"),
            bpf_filter=filter or config.get("network.filter"),
            offline_file=offline_file or config.get("network.offline_file")
        )
        
        if success:
            # 创建PID文件
            pid_file = _create_pid_file()
            if pid_file:
                print_info(f"PID文件已创建: {pid_file}")

            print_success("异常检测系统启动成功!")
            if offline_file:
                print_info(f"正在处理离线文件: {offline_file}")
            else:
                print_info(f"正在监听接口: {interface or config.get('network.interface', 'eth0')}")
                if filter or config.get("network.filter"):
                    print_info(f"使用过滤规则: {filter or config.get('network.filter')}")

            # 主循环，保持程序运行
            try:
                while _running and system_manager.is_running:
                    time.sleep(1)
            except KeyboardInterrupt:
                print_info("收到键盘中断信号，正在停止系统...")
                _running = False

            # 正常退出时停止系统
            if system_manager.is_running:
                print_info("正在停止系统...")
                system_manager.stop()

            # 删除PID文件
            _remove_pid_file()
            print_info("系统已停止")
        else:
            print_error("异常检测系统启动失败!")
            _remove_pid_file()
            raise typer.Exit(code=1)
    except Exception as e:
        print_error(f"启动系统时发生错误: {str(e)}")
        raise typer.Exit(code=1)

@app.command(name="start")
def start_command(
    interface: Optional[str] = typer.Option(
        None, "--interface", "-i",
        help="网络接口名称"
    ),
    filter: Optional[str] = typer.Option(
        None, "--filter", "-f",
        help="BPF过滤规则"
    ),
    offline_file: Optional[str] = typer.Option(
        None, "--offline-file", "-o",
        help="离线pcap文件路径"
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", "-l",
        help="日志级别"
    ),
    config_dir: str = typer.Option(
        "config", "--config-dir", "-c",
        help="配置文件目录"
    ),
    daemon: bool = typer.Option(
        False, "--daemon", "-d",
        help="以守护进程模式运行"
    ),
):
    """
    启动异常检测系统
    """
    start_system(
        config_dir=config_dir,
        daemon=daemon,
        interface=interface,
        filter=filter,
        offline_file=offline_file,
        log_level=log_level
    )

@app.command(name="stop")
def stop_command():
    """
    停止异常检测系统
    """
    print_info("正在停止异常检测系统...")
    
    try:
        # 创建系统管理器（使用默认配置）
        system_manager = SystemManager()
        
        # 停止系统
        system_manager.stop()
        
        # 删除PID文件
        _remove_pid_file()
        
        print_success("异常检测系统已停止!")
        
    except Exception as e:
        print_error(f"停止系统时发生错误: {str(e)}")
        raise typer.Exit(code=1)