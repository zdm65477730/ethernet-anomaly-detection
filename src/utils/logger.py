import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

# 日志格式配置
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs"))


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """获取指定名称的日志器"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if not logger.handlers:
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(console_handler)
    
    return logger


def setup_rotating_logger(
    name: str,
    log_file: str,
    max_bytes: int = 10 * 1024 * 1024,  # 单个日志文件最大10MB
    backup_count: int = 5,             # 保留5个备份
    level: int = logging.INFO
) -> logging.Logger:
    """设置带有轮转功能的日志器"""
    # 创建日志目录（如果不存在）
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        # 轮转文件处理器
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(file_handler)
    
    return logger


def init_logger(
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5
) -> None:
    """
    初始化全局日志系统（项目启动时调用）
    
    参数:
        log_dir: 日志存储目录（默认使用项目根目录下的logs/）
        level: 全局日志级别
        max_bytes: 单个日志文件最大字节数
        backup_count: 日志备份文件数量
    """
    # 确定日志目录
    log_dir = log_dir or DEFAULT_LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    
    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除已有的处理器（避免重复）
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root_logger.addHandler(console_handler)
    
    # 添加全局轮转文件处理器（记录所有日志）
    global_log_file = os.path.join(log_dir, "system.log")
    file_handler = RotatingFileHandler(
        global_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root_logger.addHandler(file_handler)
    
    # 记录初始化完成日志
    root_logger.info(f"日志系统初始化完成，日志目录: {log_dir}")


def set_log_level(level: int) -> None:
    """设置全局日志级别"""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)