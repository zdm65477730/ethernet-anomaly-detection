"""
启动命令模块
"""
import os
import time
import socket
import typer
from typing import Optional
from src.config.config_manager import ConfigManager
from src.system.system_manager import SystemManager
from src.utils.logger import get_logger

logger = get_logger("cli.start")

# 定义上下文设置
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

# 创建Typer应用
app = typer.Typer(help="启动异常检测系统")

def get_default_interface():
    """
    获取默认网络接口
    
    Returns:
        str: 默认网络接口名称，如果无法获取则返回None
    """
    try:
        # 创建一个UDP socket
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # 连接到一个远程地址（不需要真实存在）
            s.connect(("8.8.8.8", 80))
            # 获取本地地址
            local_ip = s.getsockname()[0]
            
        # 根据本地IP确定接口
        import netifaces
        for interface in netifaces.interfaces():
            addresses = netifaces.ifaddresses(interface)
            if netifaces.AF_INET in addresses:
                for addr_info in addresses[netifaces.AF_INET]:
                    if addr_info['addr'] == local_ip:
                        return interface
        return None
    except Exception:
        return None

def print_success(message: str):
    """打印成功消息"""
    print(f"[+] {message}")

def print_error(message: str):
    """打印错误消息"""
    print(f"[-] {message}")

def print_info(message: str):
    """打印信息消息"""
    print(f"[*] {message}")

def start_system(config: ConfigManager, interface: str = None, offline_file: str = None):
    """
    启动异常检测系统
    
    Args:
        config: 配置管理器实例
        interface: 网络接口名称
        offline_file: 离线pcap文件路径
    """
    try:
        # 初始化系统管理器
        system_manager = SystemManager(config)
        
        # 启动系统
        if offline_file:
            # 离线模式
            logger.info(f"正在启动离线模式，处理文件: {offline_file}")
            # 验证文件存在
            if not os.path.exists(offline_file):
                print_error(f"离线文件不存在: {offline_file}")
                return False
            
            # 设置离线文件
            packet_capture = system_manager.get_component("packet_capture")
            if packet_capture:
                packet_capture.set_offline_file(offline_file)
            
            # 启动系统
            success = system_manager.start_system()
        else:
            # 实时模式
            logger.info("正在启动实时模式")
            if not interface:
                # 尝试获取默认接口
                interface = get_default_interface()
                if not interface:
                    print_error("无法自动获取网络接口，请手动指定")
                    return False
            
            logger.info(f"使用网络接口: {interface}")
            success = system_manager.start_system()
        
        if not success:
            print_error("系统启动失败")
            return False
            
        # 等待系统运行
        try:
            print_info("系统正在运行，按 Ctrl+C 停止")
            pid_file = "/tmp/anomaly_detector.pid"
            
            # 写入PID文件
            with open(pid_file, "w") as f:
                f.write(str(os.getpid()))
            
            # 主循环
            while system_manager.is_running():
                # 检查离线文件处理是否完成
                packet_capture = system_manager.get_component("packet_capture")
                if packet_capture and packet_capture.offline_processing_complete:
                    logger.info("离线文件处理完成，正在停止系统")
                    break
                
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("收到中断信号")
        finally:
            # 停止系统
            logger.info("正在停止系统")
            system_manager.stop_system()
            
            # 删除PID文件
            if os.path.exists(pid_file):
                os.remove(pid_file)
            
            print_info("系统已停止")
            return True
            
    except Exception as e:
        logger.error(f"启动系统时发生错误: {e}", exc_info=True)
        print_error(f"启动系统时发生错误: {e}")
        return False

@app.command()
def main(
    interface: Optional[str] = typer.Option(None, "-i", "--interface", help="网络接口名称"),
    offline_file: Optional[str] = typer.Option(None, "-f", "--offline-file", help="离线pcap文件路径"),
    config_dir: str = typer.Option("config", "-c", "--config-dir", help="配置文件目录"),
    daemon: bool = typer.Option(False, "-d", "--daemon", help="以守护进程模式运行"),
    log_level: str = typer.Option("INFO", "-l", "--log-level", help="日志级别 (DEBUG, INFO, WARNING, ERROR)"),
    filter: Optional[str] = typer.Option(None, "--filter", help="BPF过滤器")
):
    """启动异常检测系统"""
    # 设置日志级别
    import logging
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
    
    # 加载配置
    try:
        config = ConfigManager(config_dir=config_dir)
    except Exception as e:
        print_error(f"配置加载失败: {e}")
        return False
    
    # 启动系统
    start_system(
        config=config,
        interface=interface,
        offline_file=offline_file
    )

if __name__ == '__main__':
    app()