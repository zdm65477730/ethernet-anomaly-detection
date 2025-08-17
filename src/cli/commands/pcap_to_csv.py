"""
从PCAP文件生成训练数据的CLI命令
"""

import os
import typer
import pandas as pd
import time
from src.capture.packet_capture import PacketCapture
from src.capture.session_tracker import SessionTracker
from src.features.stat_extractor import StatFeatureExtractor
from src.data.data_processor import DataProcessor
from src.cli.utils import print_success, print_error, print_info
from src.config.config_manager import ConfigManager

def pcap_to_csv(
    pcap_file: str = typer.Option(
        ..., "--pcap-file", "-f",
        help="输入的PCAP文件路径"
    ),
    output_dir: str = typer.Option(
        "data/processed", "--output", "-o",
        help="输出目录"
    ),
    config_dir: str = typer.Option(
        "config", "--config-dir", "-c",
        help="配置文件目录"
    )
):
    """
    从PCAP文件提取特征并生成用于训练的CSV数据集
    """
    try:
        # 检查PCAP文件是否存在
        if not os.path.exists(pcap_file):
            print_error(f"PCAP文件不存在: {pcap_file}")
            raise typer.Exit(code=1)
        
        # 加载配置
        config = ConfigManager(config_dir=config_dir)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        print_info(f"正在处理PCAP文件: {pcap_file}")
        print_info(f"输出目录: {output_dir}")
        
        # 初始化组件
        packet_capture = PacketCapture(offline_file=pcap_file)
        session_tracker = SessionTracker()
        data_processor = DataProcessor(config=config)
        feature_extractor = StatFeatureExtractor(config=config)
        
        # 启动组件
        session_tracker.start()
        packet_capture.start()
        
        # 等待组件启动
        time.sleep(0.1)
        
        # 处理所有数据包
        packet_count = 0
        max_wait_time = 30  # 最大等待时间30秒
        start_time = time.time()
        
        while True:
            packet_data = packet_capture.get_next_packet()
            if packet_data:
                # 解析数据包并更新会话
                try:
                    session_tracker.process_packet(packet_data)
                    packet_count += 1
                    
                    # 每处理10个包显示一次进度
                    if packet_count % 10 == 0:
                        print_info(f"已处理 {packet_count} 个数据包")
                except Exception as e:
                    print_error(f"处理数据包时出错: {str(e)}")
                    continue
            elif time.time() - start_time > max_wait_time:
                # 超时退出
                break
            else:
                # 短暂等待更多数据包
                time.sleep(0.01)
        
        packet_capture.stop()
        session_tracker.stop()
        print_info(f"总共处理了 {packet_count} 个数据包")
        
        # 显示会话统计信息
        session_count = len(session_tracker.sessions)
        print_info(f"总共创建了 {session_count} 个会话")
        
        # 从会话中提取特征
        print_info("正在提取特征...")
        features_data = []
        
        session_count = 0
        for session_id, session in session_tracker.sessions.items():
            # 从会话提取统计特征
            features = feature_extractor.extract_features_from_session(session)
            if features and any(v != 0 for v in features.values()):
                # 添加标签字段（这里简化处理，实际应用中可能需要更复杂的标签生成逻辑）
                features['label'] = 0  # 默认标记为正常流量
                features_data.append(features)
                session_count += 1
            elif features:
                print_info(f"会话 {session_id} 的特征全为零值")
            else:
                print_info(f"会话 {session_id} 未提取到特征")
        
        print_info(f"从 {session_count} 个会话中提取特征")
        
        if not features_data:
            print_error("未能从PCAP文件中提取任何特征数据")
            raise typer.Exit(code=1)
        
        # 转换为DataFrame
        df = pd.DataFrame(features_data)
        print_info(f"提取了 {len(df)} 条特征记录")
        
        # 数据预处理
        print_info("正在进行数据预处理...")
        try:
            # 清理数据
            df = data_processor.clean_data(df)
            
            # 保存预处理后的数据
            output_file = os.path.join(output_dir, "processed_data.csv")
            df.to_csv(output_file, index=False)
            print_success(f"预处理后的数据已保存至: {output_file}")
            
            # 获取模型兼容特征
            model_features, _ = data_processor.get_model_compatible_features()
            available_features = [f for f in model_features if f in df.columns]
            
            # 保存模型兼容特征数据
            if available_features:
                model_df = df[available_features + ['label']]  # 包含标签列
                model_output_file = os.path.join(output_dir, "model_features_data.csv")
                model_df.to_csv(model_output_file, index=False)
                print_success(f"模型兼容特征数据已保存至: {model_output_file}")
            
            print_success(f"成功从PCAP文件生成训练数据!")
            print_info(f"总记录数: {len(df)}")
            print_info(f"特征数量: {len(available_features) if available_features else 'N/A'}")
            
        except Exception as e:
            print_error(f"数据预处理失败: {str(e)}")
            raise typer.Exit(code=1)
        
    except Exception as e:
        print_error(f"处理PCAP文件时出错: {str(e)}")
        raise typer.Exit(code=1)

# 创建Typer应用实例
app = typer.Typer(help="从PCAP文件生成训练数据", no_args_is_help=True)

# 注册命令
@app.command("pcap-to-csv")
def pcap_to_csv_command(
    pcap_file: str = typer.Option(
        ..., "--pcap-file", "-f",
        help="输入的PCAP文件路径"
    ),
    output_dir: str = typer.Option(
        "data/processed", "--output", "-o",
        help="输出目录"
    ),
    config_dir: str = typer.Option(
        "config", "--config-dir", "-c",
        help="配置文件目录"
    )
):
    """从PCAP文件提取特征并生成用于训练的CSV数据集"""
    return pcap_to_csv(pcap_file=pcap_file, output_dir=output_dir, config_dir=config_dir)

if __name__ == "__main__":
    app()