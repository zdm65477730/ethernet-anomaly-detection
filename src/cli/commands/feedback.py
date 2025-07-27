import os
import json
import typer
from datetime import datetime
from typing import Optional
from src.cli.utils import print_info, print_success, print_error, print_warning
from src.config.config_manager import ConfigManager
from src.detection.feedback_processor import FeedbackProcessor
from src.utils.logger import setup_logging, get_logger

app = typer.Typer(help="反馈处理相关命令")

logger = get_logger("cli.feedback")

@app.command(name="submit")
def submit_feedback(
    detection_id: str = typer.Option(
        ..., "--detection-id", "-id",
        help="检测ID"
    ),
    is_anomaly: bool = typer.Option(
        ..., "--is-anomaly", "-a",
        help="是否为真实异常"
    ),
    anomaly_type: Optional[str] = typer.Option(
        None, "--anomaly-type", "-t",
        help="异常类型（如果标记为异常）"
    ),
    config_dir: str = typer.Option(
        "config", "--config-dir", "-c",
        help="配置文件目录"
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", "-l",
        help="日志级别"
    )
):
    """
    提交检测结果反馈
    """
    # 配置日志
    setup_logging(level=log_level)
    
    try:
        # 加载配置
        config = ConfigManager(config_dir=config_dir)
        
        # 创建反馈处理器实例
        feedback_processor = FeedbackProcessor(config=config)
        
        # 构造反馈数据
        feedback_data = {
            "anomaly_id": detection_id,
            "is_correct": is_anomaly,
            "actual_type": anomaly_type if is_anomaly else None,
            "timestamp": datetime.now().timestamp()
        }
        
        # 提交反馈
        success = feedback_processor.submit_feedback(feedback_data)
        
        if success:
            print_success(f"反馈提交成功: 检测ID {detection_id}")
            print_info("反馈数据已保存至 data/feedback/")
        else:
            print_error("反馈提交失败")
            raise typer.Exit(code=1)
            
    except Exception as e:
        print_error(f"提交反馈时出错: {str(e)}")
        logger.error("提交反馈时出错", exc_info=True)
        raise typer.Exit(code=1)

@app.command(name="list")
def list_feedback(
    limit: int = typer.Option(
        10, "--limit", "-n",
        help="显示反馈数量限制"
    ),
    config_dir: str = typer.Option(
        "config", "--config-dir", "-c",
        help="配置文件目录"
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", "-l",
        help="日志级别"
    )
):
    """
    列出已提交的反馈
    """
    # 配置日志
    setup_logging(level=log_level)
    
    try:
        # 加载配置
        config = ConfigManager(config_dir=config_dir)
        
        # 反馈存储目录
        feedback_dir = config.get("feedback.dir", "data/feedback")
        
        if not os.path.exists(feedback_dir):
            print_warning("反馈目录不存在")
            return
        
        # 获取反馈文件列表
        feedback_files = [f for f in os.listdir(feedback_dir) if f.endswith('.json')]
        feedback_files.sort(reverse=True)  # 按时间倒序排列
        
        if not feedback_files:
            print_info("暂无反馈数据")
            return
        
        print_info(f"找到 {len(feedback_files)} 个反馈文件")
        
        # 显示最近的反馈
        for i, filename in enumerate(feedback_files[:limit]):
            file_path = os.path.join(feedback_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    feedbacks = json.load(f)
                
                print_info(f"\n[{i+1}] 文件: {filename} (包含 {len(feedbacks)} 条反馈)")
                for j, feedback in enumerate(feedbacks[-5:]):  # 显示每文件最近5条
                    timestamp = datetime.fromtimestamp(feedback.get('timestamp', 0))
                    print_info(f"  {j+1}. ID: {feedback.get('anomaly_id', 'N/A')}, "
                              f"正确: {feedback.get('is_correct', 'N/A')}, "
                              f"时间: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception as e:
                print_error(f"读取文件 {filename} 时出错: {str(e)}")
        
    except Exception as e:
        print_error(f"列出反馈时出错: {str(e)}")
        logger.error("列出反馈时出错", exc_info=True)
        raise typer.Exit(code=1)

@app.command(name="cleanup")
def cleanup_feedback(
    days: int = typer.Option(
        30, "--days", "-d",
        help="保留天数"
    ),
    config_dir: str = typer.Option(
        "config", "--config-dir", "-c",
        help="配置文件目录"
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", "-l",
        help="日志级别"
    )
):
    """
    清理旧的反馈数据
    """
    # 配置日志
    setup_logging(level=log_level)
    
    try:
        # 加载配置
        config = ConfigManager(config_dir=config_dir)
        
        # 反馈存储目录
        feedback_dir = config.get("feedback.dir", "data/feedback")
        
        if not os.path.exists(feedback_dir):
            print_warning("反馈目录不存在")
            return
        
        # 计算保留时间戳
        from datetime import datetime, timedelta
        cutoff_time = (datetime.now() - timedelta(days=days)).timestamp()
        
        # 查找并删除旧文件
        deleted_count = 0
        feedback_files = [f for f in os.listdir(feedback_dir) if f.endswith('.json')]
        
        for filename in feedback_files:
            file_path = os.path.join(feedback_dir, filename)
            try:
                # 从文件名提取日期
                if filename.startswith('feedback_') and filename.endswith('.json'):
                    date_str = filename[9:-5]  # 提取 YYYYMMDD 部分
                    file_date = datetime.strptime(date_str, '%Y%m%d').timestamp()
                    
                    if file_date < cutoff_time:
                        os.remove(file_path)
                        print_info(f"已删除旧反馈文件: {filename}")
                        deleted_count += 1
            except Exception as e:
                print_warning(f"处理文件 {filename} 时出错: {str(e)}")
        
        print_success(f"清理完成，共删除 {deleted_count} 个旧反馈文件")
        
    except Exception as e:
        print_error(f"清理反馈时出错: {str(e)}")
        logger.error("清理反馈时出错", exc_info=True)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()