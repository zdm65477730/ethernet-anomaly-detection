import argparse
import logging
from src.system.system_manager import SystemManager
from src.utils.logger import setup_logging
from src import __version__

def main():
    """异常检测系统主入口"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description=f"以太网异常检测系统 v{__version__}")
    parser.add_argument("--interface", help="网络接口名称（如eth0）", default=None)
    parser.add_argument("--filter", help="BPF过滤规则（如'tcp port 80'）", default=None)
    parser.add_argument("--log-level", help="日志级别", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    
    args = parser.parse_args()
    
    # 初始化日志
    setup_logging(level=args.log_level)
    logger = logging.getLogger("main")
    logger.info(f"启动以太网异常检测系统 v{__version__}")
    
    try:
        # 创建并启动系统管理器
        system_manager = SystemManager()
        system_manager.start(interface=args.interface, bpf_filter=args.filter)
        
        # 保持主进程运行
        logger.info("系统启动成功，按Ctrl+C停止")
        while True:
            input("")  # 等待用户输入退出
            
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止系统...")
    except Exception as e:
        logger.error(f"系统运行出错: {str(e)}", exc_info=True)
    finally:
        if 'system_manager' in locals() and system_manager._is_running:
            system_manager.stop()
        logger.info("系统已停止")

if __name__ == "__main__":
    main()
