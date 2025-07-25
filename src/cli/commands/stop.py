import os
import typer
import time
import signal
from typing import Optional
from src.cli.utils import (
    print_success,
    print_error,
    print_info,
    confirm
)
from src.config.config_manager import ConfigManager

def main(
    pid_file: Optional[str] = typer.Option(
        None, "--pid-file", "-p",
        help="PID文件路径"
    ),
    config_dir: str = typer.Option(
        "config", "--config-dir", "-c",
        help="配置文件目录"
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="强制终止进程"
    )
):
    """停止异常检测系统"""
    # 确定PID文件路径
    if pid_file is None:
        try:
            config = ConfigManager(config_dir=config_dir)
            pid_file = config.get("system.pid_file", "anomaly_detector.pid")
        except Exception:
            pid_file = "anomaly_detector.pid"
    
    # 检查PID文件是否存在
    if not os.path.exists(pid_file):
        print_error(f"未找到PID文件: {pid_file}")
        if confirm("是否要强制搜索并停止所有相关进程?"):
            # 简单的进程搜索实现
            try:
                import psutil
                stopped = False
                for proc in psutil.process_iter(["name", "cmdline"]):
                    try:
                        if "anomaly-detector" in proc.name() or \
                           "anomaly_detector" in " ".join(proc.cmdline()):
                            print_info(f"找到进程 {proc.pid}，正在终止...")
                            proc.terminate()
                            # 等待进程终止
                            try:
                                proc.wait(timeout=5)
                                print_success(f"进程 {proc.pid} 已终止")
                                stopped = True
                            except psutil.TimeoutExpired:
                                if force:
                                    proc.kill()
                                    print_success(f"强制终止进程 {proc.pid}")
                                    stopped = True
                                else:
                                    print_error(f"进程 {proc.pid} 未能正常终止，使用--force强制终止")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                if stopped:
                    print_success("所有相关进程已停止")
                    raise typer.Exit(code=0)
                else:
                    print_info("未找到运行中的异常检测系统进程")
                    raise typer.Exit(code=0)
            except ImportError:
                print_error("未安装psutil库，无法搜索进程，请手动指定PID")
                raise typer.Exit(code=1)
        else:
            print_info("停止操作已取消")
            raise typer.Exit(code=0)
    
    # 读取PID
    try:
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())
        print_info(f"找到PID: {pid}")
    except (ValueError, IOError) as e:
        print_error(f"读取PID文件失败: {str(e)}")
        raise typer.Exit(code=1)
    
    # 检查进程是否存在
    try:
        os.kill(pid, 0)  # 发送0信号检查进程是否存在
    except OSError:
        print_error(f"进程 {pid} 不存在，删除无效PID文件")
        os.remove(pid_file)
        raise typer.Exit(code=0)
    
    # 尝试终止进程
    try:
        print_info(f"正在终止进程 {pid}...")
        os.kill(pid, signal.SIGTERM)  # 发送终止信号
        
        # 等待进程终止
        timeout = 10
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                os.kill(pid, 0)
                time.sleep(0.5)
            except OSError:
                break
        
        # 检查是否已终止
        try:
            os.kill(pid, 0)
            if force:
                print_warning(f"进程 {pid} 未能正常终止，尝试强制终止...")
                os.kill(pid, signal.SIGKILL)
                time.sleep(1)
                try:
                    os.kill(pid, 0)
                    print_error(f"强制终止进程 {pid} 失败，请手动处理")
                    raise typer.Exit(code=1)
                except OSError:
                    pass
            else:
                print_error(f"进程 {pid} 未能正常终止，使用--force强制终止")
                raise typer.Exit(code=1)
        
        # 删除PID文件
        if os.path.exists(pid_file):
            os.remove(pid_file)
        
        print_success(f"进程 {pid} 已成功终止")
    except OSError as e:
        print_error(f"终止进程失败: {str(e)}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    typer.run(main)
