import os
import time
import signal
from src.cli.utils import print_info, print_success, print_error, print_warning
import typer

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# 添加调试打印函数
def print_debug(message):
    """调试信息打印"""
    if os.environ.get('DEBUG', '').lower() in ('1', 'true', 'yes'):
        print(f"[DEBUG] {message}")

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
        print_info("执行系统停止命令...")
        if force:
            print_warning("使用强制终止模式")
        stop_system(force, pid_file)
        print_info("系统停止命令执行完成")

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
        print_info("找到主系统PID文件，正在停止主系统...")
        _stop_process_by_pid_file(main_pid_file, "主系统")
        stopped_any_process = True
        # 等待一小段时间确保进程确实停止
        time.sleep(1)
    
    # 检查持续训练PID文件
    continuous_pid_file = "continuous_training.pid"
    if os.path.exists(continuous_pid_file):
        print_info("找到持续训练PID文件，正在停止持续训练...")
        _stop_process_by_pid_file(continuous_pid_file, "持续训练")
        stopped_any_process = True
        # 等待一小段时间确保进程确实停止
        time.sleep(1)
    
    # 检查自驱动训练PID文件
    self_driving_pid_file = "self_driving_training.pid"
    if os.path.exists(self_driving_pid_file):
        print_info("找到自驱动训练PID文件，正在停止自驱动训练...")
        _stop_process_by_pid_file(self_driving_pid_file, "自驱动训练")
        stopped_any_process = True
        # 等待一小段时间确保进程确实停止
        time.sleep(1)
    
    # 如果没有找到PID文件，尝试通过进程名查找
    if not stopped_any_process:
        if PSUTIL_AVAILABLE:
            try:
                stopped_pids = []  # 记录已停止的PID，避免重复处理
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        # 检查进程命令行是否包含anomaly-detector
                        if proc.info['cmdline'] and any('anomaly-detector' in arg for arg in proc.info['cmdline']):
                            pid = proc.info['pid']
                            # 避免重复处理同一个PID
                            if pid in stopped_pids:
                                continue
                                
                            print_info(f"发现运行中的异常检测系统进程 (PID: {pid})")
                            if force or _send_termination_signal(pid, force):
                                stopped_any_process = True
                                stopped_pids.append(pid)
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        # 进程可能已经退出或无法访问
                        pass
            except Exception as e:
                print_warning(f"查找进程时出错: {e}")
        else:
            print_warning("未安装psutil库，无法通过进程名查找系统进程")
    
    # 如果仍然没有停止任何进程，尝试查找所有相关的python进程
    if not stopped_any_process:
        if PSUTIL_AVAILABLE:
            try:
                stopped_pids = []  # 记录已停止的PID，避免重复处理
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        # 检查进程命令行是否包含相关路径
                        if proc.info['cmdline'] and any('train' in arg and 'self-driving' in arg for arg in proc.info['cmdline']):
                            pid = proc.info['pid']
                            # 避免重复处理同一个PID
                            if pid in stopped_pids:
                                continue
                                
                            print_info(f"发现自驱动训练进程 (PID: {pid})")
                            if force or _send_termination_signal(pid, force):
                                stopped_any_process = True
                                stopped_pids.append(pid)
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        # 进程可能已经退出或无法访问
                        pass
            except Exception as e:
                print_warning(f"查找自驱动训练进程时出错: {e}")
    
    if not stopped_any_process:
        print_warning("系统似乎未运行（未找到PID文件或相关进程）")
        return

def _send_termination_signal(pid: int, force: bool = False) -> bool:
    """发送终止信号到进程"""
    signal_type = signal.SIGKILL if force else signal.SIGTERM
    signal_name = "SIGKILL(强制)" if force else "SIGTERM"
    
    try:
        os.kill(pid, signal_type)
        if force:
            print_info(f"已发送强制终止信号到进程 (PID: {pid})")
        else:
            print_info(f"已发送停止信号到进程 (PID: {pid})")
        return True
    except OSError as e:
        if e.errno == 1:  # Operation not permitted
            print_error(f"发送{signal_name}信号到进程失败: 权限不足")
            print_info("提示: 系统可能是使用sudo启动的，请使用sudo运行stop命令")
        else:
            print_error(f"发送{signal_name}信号到进程失败: {e}")
        return False

def _stop_process_by_pid_file(pid_file: str, process_name: str):
    """根据PID文件停止进程"""
    # 读取PID
    try:
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())
        print_debug(f"从PID文件读取到PID: {pid}")
    except (ValueError, IOError) as e:
        print_warning(f"读取{process_name} PID文件失败: {e}")
        return False
    
    # 检查进程是否存在
    try:
        os.kill(pid, 0)  # 检查进程是否存在
        process_exists = True
        print_debug(f"进程 {pid} 存在")
    except OSError as e:
        # 如果是权限错误，我们假设进程存在（因为它是由sudo启动的）
        if e.errno == 1:  # Operation not permitted
            process_exists = True
            print_debug(f"无法检查进程 {pid} 是否存在 (权限不足)，假设进程存在")
        else:
            process_exists = False
            print_debug(f"进程 {pid} 不存在: {e}")
    
    if not process_exists:
        print_warning(f"{process_name}进程似乎已退出")
        # 删除无效的PID文件
        try:
            os.remove(pid_file)
        except:
            pass
        return True  # 进程已退出，视为成功处理
    
    print_info(f"找到{process_name}进程 (PID: {pid})，正在发送停止信号...")
    
    # 发送终止信号
    if not _send_termination_signal(pid, False):  # 默认不强制
        return False
    
    # 等待进程退出
    timeout = 20  # 增加到20秒超时
    start_time = time.time()
    process_stopped = False
    while time.time() - start_time < timeout:
        try:
            os.kill(pid, 0)  # 检查进程是否还存在
            print_debug(f"进程 {pid} 仍在运行，继续等待... (已等待 {time.time() - start_time:.1f} 秒)")
            time.sleep(1)
        except OSError as e:
            # 如果是权限错误，说明进程可能仍在运行，继续检查
            if e.errno == 1:  # Operation not permitted
                print_debug(f"无法检查进程 {pid} 状态 (权限不足)，继续等待...")
                time.sleep(1)
            else:
                # 进程已退出
                print_info(f"{process_name}进程已退出")
                process_stopped = True
                break
    
    # 如果进程已停止，清理PID文件
    if process_stopped:
        try:
            if os.path.exists(pid_file):
                os.remove(pid_file)
            print_success(f"{process_name}已停止!")
            return True
        except Exception as e:
            print_warning(f"删除{process_name} PID文件失败: {e}")
            print_success(f"{process_name}已停止!")
            return True
    
    # 进程未能在超时时间内正常退出，尝试强制终止
    print_warning(f"{process_name}进程未能在超时时间内正常退出")
    if _send_termination_signal(pid, True):  # 强制终止
        # 再次等待
        start_time = time.time()
        force_stopped = False
        while time.time() - start_time < 10:
            try:
                os.kill(pid, 0)
                print_debug(f"强制终止后进程 {pid} 仍在运行，继续等待... (已等待 {time.time() - start_time:.1f} 秒)")
                time.sleep(1)
            except OSError as e:
                # 如果是权限错误，说明进程可能仍在运行，继续检查
                if e.errno == 1:  # Operation not permitted
                    print_debug(f"无法检查进程 {pid} 状态 (权限不足)，继续等待...")
                    time.sleep(1)
                else:
                    # 进程已退出
                    force_stopped = True
                    break
        
        if force_stopped:
            try:
                if os.path.exists(pid_file):
                    os.remove(pid_file)
                print_success(f"{process_name}已停止!")
                return True
            except Exception as e:
                print_warning(f"删除{process_name} PID文件失败: {e}")
                print_success(f"{process_name}已停止!")
                return True
        else:
            print_error(f"无法强制终止{process_name}进程 (PID: {pid})")
            print_info("提示: 可能需要使用sudo权限来终止该进程")
            return False
    else:
        print_error(f"无法发送强制终止信号到{process_name}进程 (PID: {pid})")
        print_info("提示: 可能需要使用sudo权限来终止该进程")
        return False

def is_process_running(pid):
    """检查指定PID的进程是否正在运行"""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def terminate_process(pid, timeout=10):
    """终止指定PID的进程"""
    try:
        # 首先尝试正常终止
        os.kill(pid, signal.SIGTERM)
        print_info(f"已发送SIGTERM信号到进程 {pid}")
        
        # 等待进程退出
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not is_process_running(pid):
                print_success(f"进程 {pid} 已正常退出")
                return True
            time.sleep(0.5)
        
        # 如果进程仍未退出，强制终止
        print_warning(f"进程 {pid} 未能在 {timeout} 秒内正常退出，正在强制终止...")
        os.kill(pid, signal.SIGKILL)
        print_info(f"已发送SIGKILL信号到进程 {pid}")
        
        # 再次等待进程退出
        start_time = time.time()
        while time.time() - start_time < 5:
            if not is_process_running(pid):
                print_success(f"进程 {pid} 已被强制终止")
                return True
            time.sleep(0.1)
        
        print_error(f"无法终止进程 {pid}")
        return False
        
    except OSError as e:
        print_error(f"终止进程 {pid} 失败: {e}")
        return False