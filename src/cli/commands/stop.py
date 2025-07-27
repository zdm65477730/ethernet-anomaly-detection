import os
import time
import signal
from src.cli.utils import print_info, print_success, print_error, print_warning
import typer

app = typer.Typer(help="停止系统命令", invoke_without_command=True)

@app.callback()
def main(
    ctx: typer.Context,
    force: bool = typer.Option(False, "--force", "-f", help="强制终止进程"),
    pid_file: str = typer.Option("system.pid", "--pid-file", help="PID文件路径")
):
    """
    停止异常检测系统
    """
    # 如果是直接调用stop命令而不是子命令，则执行停止
    if ctx.invoked_subcommand is None:
        stop_system(force, pid_file)

def stop_system(force: bool = False, pid_file: str = "system.pid"):
    """
    停止异常检测系统
    
    Args:
        force: 是否强制终止进程
        pid_file: PID文件路径
    """
    print_info("正在停止异常检测系统...")
    
    stopped_any_process = False
    
    # 检查主系统PID文件
    main_pid_file = "anomaly_detector.pid"
    if os.path.exists(main_pid_file):
        stopped_any_process = True
        _stop_process_by_pid_file(main_pid_file, "主系统")
    
    # 检查持续训练PID文件
    continuous_pid_file = "continuous_training.pid"
    if os.path.exists(continuous_pid_file):
        stopped_any_process = True
        _stop_process_by_pid_file(continuous_pid_file, "持续训练")
    
    if not stopped_any_process:
        print_warning("系统似乎未运行（未找到PID文件）")
        return

def _stop_process_by_pid_file(pid_file: str, process_name: str):
    """根据PID文件停止进程"""
    # 读取PID
    try:
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())
    except (ValueError, IOError) as e:
        print_warning(f"读取{process_name} PID文件失败: {e}")
        return
    
    # 检查进程是否存在
    try:
        os.kill(pid, 0)  # 检查进程是否存在
    except OSError:
        print_warning(f"{process_name}进程似乎已退出")
        # 删除无效的PID文件
        try:
            os.remove(pid_file)
        except:
            pass
        return
    
    # 发送SIGTERM信号停止进程
    try:
        os.kill(pid, signal.SIGTERM)  # SIGTERM
        print_info(f"已发送停止信号到{process_name}进程 (PID: {pid})")
    except OSError as e:
        print_error(f"发送停止信号到{process_name}进程失败: {e}")
        return
    
    # 等待进程退出
    timeout = 10  # 10秒超时
    start_time = time.time()
    while os.path.exists(pid_file) and time.time() - start_time < timeout:
        try:
            os.kill(pid, 0)  # 检查进程是否还存在
            time.sleep(0.5)
        except OSError:
            # 进程已退出
            break
    
    # 检查进程是否已退出
    if os.path.exists(pid_file):
        try:
            os.kill(pid, 0)
            print_warning(f"{process_name}进程未能在超时时间内正常退出")
        except OSError:
            # 进程已退出，但PID文件未清理
            try:
                os.remove(pid_file)
            except:
                pass
            print_success(f"{process_name}已停止!")
    else:
        print_success(f"{process_name}已停止!")

def is_process_running(pid):
    """检查指定PID的进程是否正在运行"""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def terminate_process(pid, timeout=10):
    """优雅地终止进程"""
    try:
        import psutil
        process = psutil.Process(pid)
        process.terminate()
        
        # 等待进程终止
        start_time = time.time()
        while process.is_running() and time.time() - start_time < timeout:
            time.sleep(0.5)
        
        return not process.is_running()
    except ImportError:
        print_warning("未安装psutil库，使用基础方法终止进程")
        try:
            os.kill(pid, signal.SIGTERM)
            return True
        except OSError:
            return False
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return True
    except Exception as e:
        print_warning(f"终止进程 {pid} 时出错: {e}")
        return False

# 保持向后兼容性
stop_command = main

if __name__ == "__main__":
    app()
