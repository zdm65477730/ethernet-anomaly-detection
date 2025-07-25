import os
import sys
import shutil
from typing import List, Optional, Tuple

# 终端颜色代码
COLOR_RESET = "\033[0m"
COLOR_GREEN = "\033[32m"
COLOR_RED = "\033[31m"
COLOR_YELLOW = "\033[33m"
COLOR_BLUE = "\033[34m"
COLOR_CYAN = "\033[36m"

def print_success(message: str) -> None:
    """打印成功消息（绿色）"""
    print(f"{COLOR_GREEN}[+] {message}{COLOR_RESET}")

def print_error(message: str) -> None:
    """打印错误消息（红色）"""
    print(f"{COLOR_RED}[-] {message}{COLOR_RESET}", file=sys.stderr)

def print_warning(message: str) -> None:
    """打印警告消息（黄色）"""
    print(f"{COLOR_YELLOW}[!] {message}{COLOR_RESET}")

def print_info(message: str) -> None:
    """打印信息消息（蓝色）"""
    print(f"{COLOR_BLUE}[*] {message}{COLOR_RESET}")

def print_debug(message: str) -> None:
    """打印调试消息（青色）"""
    print(f"{COLOR_CYAN}[#] {message}{COLOR_RESET}")

def confirm(prompt: str, default: bool = False) -> bool:
    """
    询问用户确认
    
    Args:
        prompt: 提示消息
        default: 默认值（True为yes，False为no）
        
    Returns:
        用户是否确认
    """
    choices = " [Y/n] " if default else " [y/N] "
    while True:
        response = input(prompt + choices).strip().lower()
        if not response:
            return default
        if response in ("y", "yes"):
            return True
        if response in ("n", "no"):
            return False
        print_warning("请输入 'y' 或 'n'")

def create_directories(dirs: List[str], overwrite: bool = False) -> Tuple[bool, List[str]]:
    """
    创建目录列表
    
    Args:
        dirs: 要创建的目录路径列表
        overwrite: 如果目录已存在，是否删除并重新创建
        
    Returns:
        (是否全部成功, 失败的目录列表)
    """
    failed = []
    for dir_path in dirs:
        try:
            if os.path.exists(dir_path):
                if overwrite:
                    shutil.rmtree(dir_path)
                    os.makedirs(dir_path, exist_ok=True)
                    print_info(f"已重新创建目录: {dir_path}")
                else:
                    print_info(f"目录已存在: {dir_path}")
            else:
                os.makedirs(dir_path, exist_ok=True)
                print_success(f"已创建目录: {dir_path}")
        except Exception as e:
            print_error(f"创建目录 {dir_path} 失败: {str(e)}")
            failed.append(dir_path)
    
    return len(failed) == 0, failed

def get_available_interfaces() -> List[str]:
    """
    获取可用的网络接口列表
    
    Returns:
        网络接口名称列表
    """
    try:
        import pcapy
        return pcapy.findalldevs()
    except ImportError:
        print_error("获取网络接口失败：未安装pcapy库")
        return []
    except Exception as e:
        print_error(f"获取网络接口失败：{str(e)}")
        return []

def validate_file_path(path: str, create_parent: bool = False) -> bool:
    """
    验证文件路径是否有效
    
    Args:
        path: 文件路径
        create_parent: 是否创建父目录
        
    Returns:
        路径是否有效
    """
    parent_dir = os.path.dirname(path)
    if not parent_dir:
        return True  # 当前目录
    
    if not os.path.exists(parent_dir):
        if create_parent:
            try:
                os.makedirs(parent_dir, exist_ok=True)
                return True
            except Exception as e:
                print_error(f"创建父目录 {parent_dir} 失败: {str(e)}")
                return False
        return False
        
    if not os.path.isdir(parent_dir):
        print_error(f"{parent_dir} 不是目录")
        return False
        
    return True
