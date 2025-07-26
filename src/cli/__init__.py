"""命令行接口模块，提供系统交互功能"""
from src.cli.main import app

# 导出主应用作为模块的可调用入口
main = app

from src.cli.utils import (
    print_success,
    print_error,
    print_warning,
    print_info,
    confirm,
    create_directories
)

__all__ = [
    "app",
    "print_success",
    "print_error",
    "print_warning",
    "print_info",
    "confirm",
    "create_directories"
]
