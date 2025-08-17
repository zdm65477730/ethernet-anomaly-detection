"""自动生成测试数据的CLI命令
"""

import os
import json
import typer
from src.data.data_generator import DataGenerator
from src.cli.utils import print_success, print_error, print_info
from src.config.config_manager import ConfigManager

# 创建一个简单的CLI函数
def generate_test_data(
    count: int = typer.Option(
        1000, "--count", "-c",
        help="生成的样本数量"
    ),
    output_dir: str = typer.Option(
        None, "--output", "-o",
        help="输出目录"
    ),
    config_dir: str = typer.Option(
        "config", "--config-dir", "-C",
        help="配置文件目录"
    ),
    pcap: bool = typer.Option(
        False, "--pcap", "-p",
        help="是否生成PCAP文件"
    ),
    anomaly_types: str = typer.Option(
        None, "--anomaly-types", "-a",
        help="自定义异常类型及占比，JSON格式字符串，例如：'{\"normal\": 0.8, \"syn_flood\": 0.1, \"port_scan\": 0.1}'"
    )
):
    """
    自动生成测试数据集
    """
    # 加载配置
    try:
        config = ConfigManager(config_dir=config_dir)
    except Exception as e:
        print_error(f"加载配置失败: {str(e)}")
        raise typer.Exit(code=1)
    
    # 如果未指定输出目录，则使用配置中的目录
    if output_dir is None:
        output_dir = config.get("data.test_dir", "data/test")
    
    print_info(f"正在生成 {count} 个测试数据样本...")
    
    # 解析自定义异常类型
    custom_anomaly_types = None
    if anomaly_types:
        try:
            custom_anomaly_types = json.loads(anomaly_types)
            print_info(f"使用自定义异常类型分布: {custom_anomaly_types}")
        except json.JSONDecodeError as e:
            print_error(f"解析自定义异常类型失败: {str(e)}")
            raise typer.Exit(code=1)
    
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成数据
        generator = DataGenerator(custom_anomaly_types=custom_anomaly_types)
        result = generator.generate(
            num_samples=count,
            output_dir=output_dir,
            split_train_test=False,  # 不需要再分割训练集和测试集
            generate_model_features=True,  # 生成模型兼容特征
            generate_pcap=pcap  # 是否生成PCAP文件
        )
        
        print_success(f"测试数据生成完成! 已保存至: {result['full_path']}")
        if pcap:
            pcap_file = os.path.join(output_dir, "simulated_traffic.pcap")
            if os.path.exists(pcap_file):
                print_success(f"PCAP文件已生成: {pcap_file}")
            else:
                print_error("PCAP文件生成失败")
        
    except Exception as e:
        print_error(f"生成测试数据时出错: {str(e)}")
        raise typer.Exit(code=1)

app = typer.Typer(help="生成测试数据", no_args_is_help=True)
app.command()(generate_test_data)

if __name__ == "__main__":
    app()