import os
import typer
import time
from typing import Optional, List
from src.cli.utils import (
    print_success,
    print_error,
    print_info,
    print_warning,
    confirm
)
from src.training.continuous_trainer import ContinuousTrainer
from src.training.model_trainer import ModelTrainer
from src.data.data_processor import DataProcessor
from src.models.model_factory import ModelFactory
from src.config.config_manager import ConfigManager
from src.utils.logger import setup_logging

# 创建子命令
train_app = typer.Typer(
    name="train",
    help="模型训练相关操作"
)

@train_app.command(name="once", help="执行单次模型训练")
def train_once(
    model_type: str = typer.Option(
        "xgboost", "--model", "-m",
        help="模型类型 (xgboost, random_forest, lstm)"
    ),
    data_path: Optional[str] = typer.Option(
        None, "--data", "-d",
        help="训练数据路径，不指定则使用默认数据目录"
    ),
    test_size: float = typer.Option(
        0.2, "--test-size", "-t",
        help="测试集比例"
    ),
    cv_folds: Optional[int] = typer.Option(
        None, "--cv", "-k",
        help="交叉验证折数"
    ),
    config_dir: str = typer.Option(
        "config", "--config-dir", "-c",
        help="配置文件目录"
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="模型输出目录，不指定则使用默认目录"
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", "-l",
        help="日志级别 (DEBUG, INFO, WARNING, ERROR)"
    )
):
    """执行单次模型训练"""
    # 配置日志
    setup_logging(log_level=log_level)
    
    # 加载配置
    try:
        config = ConfigManager(config_dir=config_dir)
    except Exception as e:
        print_error(f"加载配置失败: {str(e)}")
        raise typer.Exit(code=1)
    
    # 确定交叉验证折数
    if cv_folds is None:
        cv_folds = config.get("training.cross_validation_folds", 5)
    
    # 初始化组件
    data_processor = DataProcessor()
    model_factory = ModelFactory()
    
    # 确定数据路径
    if not data_path:
        data_path = os.path.join(config.get("data.processed_dir", "data/processed"))
    
    if not os.path.exists(data_path):
        print_error(f"数据路径不存在: {data_path}")
        if confirm("是否要生成模拟训练数据?"):
            print_info("正在生成模拟训练数据...")
            from src.data.data_generator import DataGenerator
            generator = DataGenerator()
            generator.generate(num_samples=10000, output_dir=data_path)
            print_success(f"已生成模拟数据到 {data_path}")
        else:
            raise typer.Exit(code=1)
    
    # 加载训练数据
    print_info(f"从 {data_path} 加载训练数据...")
    try:
        X, y = data_processor.load_processed_data(data_path)
        print_info(f"加载完成，样本数: {len(X)}, 特征数: {X.shape[1]}")
    except Exception as e:
        print_error(f"加载数据失败: {str(e)}")
        raise typer.Exit(code=1)
    
    # 初始化训练器
    trainer = ModelTrainer(
        model_factory=model_factory,
        config=config
    )
    
    # 执行训练
    print_info(f"开始训练 {model_type} 模型...")
    try:
        model, metrics, report_path = trainer.train_new_model(
            model_type=model_type,
            X=X,
            y=y,
            test_size=test_size,
            cv_folds=cv_folds,
            output_dir=output_dir
        )
        
        print_success("模型训练完成!")
        print_info("评估指标:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print_info(f"详细报告已保存至: {report_path}")
        
    except Exception as e:
        print_error(f"训练失败: {str(e)}")
        raise typer.Exit(code=1)

@train_app.command(name="continuous", help="启动持续训练模式")
def train_continuous(
    interval: Optional[int] = typer.Option(
        None, "--interval", "-i",
        help="检查新数据的时间间隔(秒)"
    ),
    min_samples: Optional[int] = typer.Option(
        None, "--min-samples", "-m",
        help="触发训练的最小样本数"
    ),
    background: bool = typer.Option(
        False, "--background", "-b",
        help="后台运行模式"
    ),
    config_dir: str = typer.Option(
        "config", "--config-dir", "-c",
        help="配置文件目录"
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", "-l",
        help="日志级别 (DEBUG, INFO, WARNING, ERROR)"
    )
):
    """启动持续训练模式，定期检查新数据并增量更新模型"""
    # 配置日志
    setup_logging(log_level=log_level)
    
    # 加载配置
    try:
        config = ConfigManager(config_dir=config_dir)
    except Exception as e:
        print_error(f"加载配置失败: {str(e)}")
        raise typer.Exit(code=1)
    
    # 确定参数
    check_interval = interval or config.get("training.check_interval", 3600)
    min_samples = min_samples or config.get("training.min_samples", 1000)
    
    # 处理后台运行
    if background:
        try:
            # 简单的后台运行实现
            pid = os.fork()
            if pid > 0:
                print_success(f"持续训练已在后台启动 (PID: {pid})")
                raise typer.Exit(code=0)
        except OSError as e:
            print_error(f"无法在后台运行: {str(e)}")
            raise typer.Exit(code=1)
    
    # 初始化持续训练器
    trainer = ContinuousTrainer(config=config)
    
    # 启动持续训练
    print_info(f"启动持续训练模式，检查间隔: {check_interval}秒，最小样本数: {min_samples}")
    try:
        trainer.start(
            check_interval=check_interval,
            min_samples=min_samples
        )
        
        # 保持运行
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print_info("\n用户中断，正在停止持续训练...")
            trainer.stop()
        
        print_success("持续训练已停止")
        
    except Exception as e:
        print_error(f"持续训练失败: {str(e)}")
        trainer.stop()
        raise typer.Exit(code=1)

@train_app.command(name="evaluate", help="评估现有模型")
def evaluate_model(
    model_path: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="模型文件路径，不指定则使用最新模型"
    ),
    model_type: str = typer.Option(
        "xgboost", "--type", "-t",
        help="模型类型 (xgboost, random_forest, lstm)"
    ),
    test_data: Optional[str] = typer.Option(
        None, "--data", "-d",
        help="测试数据路径，不指定则使用默认测试集"
    ),
    config_dir: str = typer.Option(
        "config", "--config-dir", "-c",
        help="配置文件目录"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="评估报告输出路径"
    )
):
    """评估现有模型性能"""
    # 加载配置
    try:
        config = ConfigManager(config_dir=config_dir)
    except Exception as e:
        print_error(f"加载配置失败: {str(e)}")
        raise typer.Exit(code=1)
    
    # 初始化组件
    model_factory = ModelFactory()
    data_processor = DataProcessor()
    trainer = ModelTrainer(
        model_factory=model_factory,
        config=config
    )
    
    # 加载模型
    try:
        if model_path:
            model = model_factory.load_model(model_type, model_path)
        else:
            model = model_factory.load_latest_model(model_type)
            model_path = model_factory.get_latest_model_path(model_type)
        print_info(f"已加载模型: {model_path}")
    except Exception as e:
        print_error(f"加载模型失败: {str(e)}")
        raise typer.Exit(code=1)
    
    # 加载测试数据
    try:
        if test_data:
            X_test, y_test = data_processor.load_processed_data(test_data)
        else:
            test_dir = config.get("data.test_dir", "data/test")
            if not os.path.exists(test_dir):
                print_warning(f"测试数据目录 {test_dir} 不存在，使用训练数据的一部分作为测试集")
                train_dir = config.get("data.processed_dir", "data/processed")
                X, y = data_processor.load_processed_data(train_dir)
                X_test, y_test = data_processor.split_train_test(X, y, test_size=0.2)
            else:
                X_test, y_test = data_processor.load_processed_data(test_dir)
        
        print_info(f"已加载测试数据，样本数: {len(X_test)}")
    except Exception as e:
        print_error(f"加载测试数据失败: {str(e)}")
        raise typer.Exit(code=1)
    
    # 执行评估
    try:
        metrics, report_path = trainer.evaluate_model(
            model=model,
            model_type=model_type,
            X_test=X_test,
            y_test=y_test,
            output_path=output
        )
        
        print_success("模型评估完成!")
        print_info("评估指标:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print_info(f"评估报告已保存至: {report_path}")
        
    except Exception as e:
        print_error(f"评估失败: {str(e)}")
        raise typer.Exit(code=1)

def main():
    """模型训练相关操作"""
    train_app()

if __name__ == "__main__":
    main()
