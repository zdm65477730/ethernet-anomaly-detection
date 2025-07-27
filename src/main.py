import os
import sys
import logging

# 在任何其他导入之前设置环境变量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 设置Python日志级别
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# 在最早的阶段重定向stderr来屏蔽特定的日志信息
original_stderr = sys.stderr

class FilteredStderr:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        # 需要屏蔽的日志关键词
        self.skip_patterns = [
            'tensorflow',
            'cuda', 
            'cudnn',
            'cufft',
            'cublas',
            'absl',
            'computation_placer',
            'oneDNN',
            'stream_executor',
            'external/local_xla'
        ]
    
    def write(self, msg):
        # 检查是否包含需要屏蔽的模式
        msg_lower = msg.lower()
        for pattern in self.skip_patterns:
            if pattern in msg_lower:
                return  # 直接返回，不写入stderr
        # 如果不包含屏蔽模式，则正常输出
        self.original_stderr.write(msg)
    
    def flush(self):
        self.original_stderr.flush()

# 立即应用过滤器
sys.stderr = FilteredStderr(original_stderr)

import argparse
import logging as py_logging
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
    logger = py_logging.getLogger("main")
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
        if 'system_manager' in locals():
            system_manager.stop()
    except Exception as e:
        logger.error(f"系统运行出错: {str(e)}", exc_info=True)
        if 'system_manager' in locals():
            system_manager.stop()
        sys.exit(1)

if __name__ == "__main__":
    main()