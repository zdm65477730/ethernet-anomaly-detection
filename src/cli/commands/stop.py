import os
import signal
import time
import psutil
import typer
from src.cli.utils import print_success, print_error, print_info, print_warning

app = typer.Typer(help="停止运行中的系统进程")

def is_process_running(pid: int) -> bool:
    """检查指定PID的进程是否正在运行"""
    try:
        process = psutil.Process(pid)
        return process.is_running()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False

def terminate_process(pid: int, timeout: int = 10) -> bool:
    """
    优雅地终止进程
    
    Args:
        pid: 进程ID
        timeout: 等待超时时间(秒)
        
    Returns:
        是否成功终止
    """
    try:
        process = psutil.Process(pid)
        
        # 先尝试优雅终止
        process.terminate()
        
        # 等待进程结束
        try:
            process.wait(timeout=timeout)
            return True
        except psutil.TimeoutExpired:
            # 超时则强制终止
            process.kill()
            try:
                process.wait(timeout=3)
                return True
            except psutil.TimeoutExpired:
                return False
                
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        print_warning(f"无法访问进程 {pid}: {str(e)}")
        return False

@app.command()
def main(
    force: bool = typer.Option(False, "--force", "-f", help="强制终止进程"),
    pid_file: str = typer.Option("system.pid", "--pid-file", help="PID文件路径")
):
    """
    停止运行中的系统进程
    """
    try:
        # 检查PID文件是否存在
        if not os.path.exists(pid_file):
            print_error(f"PID文件不存在: {pid_file}")
            print_info("系统可能未运行或已异常终止")
            raise typer.Exit(code=1)
        
        # 读取PID
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())
        
        # 检查进程是否存在
        if not is_process_running(pid):
            print_warning(f"进程 {pid} 不存在或已终止")
            print_info("清理PID文件")
            os.remove(pid_file)
            raise typer.Exit(code=0)
        
        print_info(f"正在终止进程 {pid}...")
        
        # 终止进程
        if force:
            # 强制终止
            try:
                process = psutil.Process(pid)
                process.kill()
                print_info("已发送强制终止信号")
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print_error(f"强制终止进程失败: {str(e)}")
                raise typer.Exit(code=1)
        else:
            # 优雅终止
            if not terminate_process(pid):
                print_error(f"进程 {pid} 未能正常终止，使用--force强制终止")
                raise typer.Exit(code=1)
        
        # 删除PID文件
        if os.path.exists(pid_file):
            os.remove(pid_file)
        
        print_success(f"进程 {pid} 已成功终止")
    except OSError as e:
        print_error(f"终止进程失败: {str(e)}")
        raise typer.Exit(code=1)
    except ValueError as e:
        print_error(f"PID文件格式错误: {str(e)}")
        raise typer.Exit(code=1)
    except Exception as e:
        print_error(f"未知错误: {str(e)}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    typer.run(main)