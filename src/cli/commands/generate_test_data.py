"""
自动生成测试数据的CLI命令
"""

import os
import typer
from src.data.data_generator import DataGenerator
from src.cli.utils import print_success, print_error, print_info
from src.config.config_manager import ConfigManager

app = typer.Typer(help="生成测试数据", invoke_without_command=True)

@app.callback()
def main(
    ctx: typer.Context,
    samples: int = typer.Option(
        1000, "--samples", "-s",
        help="生成的样本数量"
    ),
    output_dir: str = typer.Option(
        "data/test", "--output", "-o",
        help="输出目录"
    ),
    config_dir: str = typer.Option(
        "config", "--config-dir", "-d",
        help="配置文件目录"
    )
):
    """
    自动生成测试数据集
    """
    # 如果是直接调用命令而不是子命令，则执行生成测试数据
    if ctx.invoked_subcommand is None:
        
        print_info(f"正在生成 {samples} 个测试数据样本...")
        
        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成数据
            generator = DataGenerator()
            result = generator.generate(
                num_samples=samples,
                output_dir=output_dir,
                split_train_test=False,  # 不需要再分割训练集和测试集
                generate_model_features=True  # 生成模型兼容特征
            )
            
            print_success(f"测试数据生成完成! 已保存至: {result['full_path']}")
            
        except Exception as e:
            print_error(f"生成测试数据时出错: {str(e)}")
            raise typer.Exit(code=1)

if __name__ == "__main__":
    app()