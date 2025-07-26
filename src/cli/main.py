import typer
from typing import Optional
from src.cli.commands import init as init_cmd
from src.cli.commands import start as start_cmd
from src.cli.commands import stop as stop_cmd
from src.cli.commands import status as status_cmd
from src.cli.commands import train as train_cmd
from src.cli.utils import print_info, print_error

# 创建Typer应用
app = typer.Typer(
    name="anomaly-detector",
    help="实时以太网异常检测系统",
    add_completion=False
)

# 添加子命令
app.command(name="init", help="初始化系统配置和目录结构")(init_cmd.main)
app.command(name="start", help="启动异常检测系统")(start_cmd.main)
app.command(name="stop", help="停止异常检测系统")(stop_cmd.main)
app.command(name="status", help="查看系统运行状态")(status_cmd.main)

# 添加train子应用
app.add_typer(train_cmd.train_app)

@app.callback()
def main_callback(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", 
        help="显示系统版本",
        is_eager=True
    )
):
    """主回调函数，处理全局选项"""
    if version:
        from src import __version__
        print_info(f"异常检测系统版本: {__version__}")
        raise typer.Exit()

if __name__ == "__main__":
    app()
