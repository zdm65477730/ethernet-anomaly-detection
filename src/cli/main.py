try:
    from src.utils.tf_silence import aggressive_silence
    aggressive_silence()
except Exception:
    pass  # 忽略任何导入错误

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
            'external/local_xla',
            'eigen',
            'registering factory',
            'attempting to register',
            'please check linkage'
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

import typer
from typing import Optional
from src.cli.commands.train import train_app
from src.cli.commands.init import app as init_app
from src.cli.commands.generate_test_data import app as generate_test_data_app
from src.cli.commands.status import main as status_command
from src.cli.commands.start import app as start_app
from src.cli.commands.stop import app as stop_app
from src.cli.commands.report import app as report_app
from src.cli.commands.feedback import app as feedback_app
from src.cli.utils import print_info, print_error

# 创建Typer应用
app = typer.Typer(
    name="anomaly-detector",
    help="实时以太网异常检测系统",
    add_completion=False
)

# 添加子命令
app.add_typer(init_app, name="init", help="系统初始化命令")
app.add_typer(generate_test_data_app, name="generate-test-data", help="生成测试数据命令")
app.add_typer(train_app, name="train", help="模型训练命令")
app.add_typer(start_app, name="start", help="启动系统命令")
app.add_typer(stop_app, name="stop", help="停止系统命令")
app.add_typer(report_app, name="report", help="检测报告命令")
app.add_typer(feedback_app, name="feedback", help="反馈处理命令")

# 添加简单命令
app.command(name="status")(status_command)

@app.callback()
def main_callback(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", 
        help="显示系统版本",
        is_eager=True
    )
):
    """主回调函数"""
    if version:
        print_info("异常检测系统 v1.0.0")
        raise typer.Exit()

if __name__ == "__main__":
    app()