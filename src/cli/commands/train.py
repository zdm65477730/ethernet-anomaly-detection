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
from src.training.feedback_optimizer import FeedbackOptimizer
from src.training.automl_trainer import AutoMLTrainer
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
    ),
    auto_optimize: bool = typer.Option(
        False, "--auto-optimize", "-a",
        help="训练后自动优化模型和特征工程"
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
        print_info(f"加载完成，样本数: {len(X)}, 特征数: {X.shape[1] if hasattr(X, 'shape') else 'unknown'}")
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
        
        # 如果启用了自动优化
        if auto_optimize:
            print_info("开始自动优化...")
            _perform_auto_optimization(
                model=model,
                model_type=model_type,
                metrics=metrics,
                config=config,
                model_factory=model_factory
            )
        
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
    # 初始化日志
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
    ),
    auto_optimize: bool = typer.Option(
        False, "--auto-optimize", "-a",
        help="评估后自动优化模型和特征工程"
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
        
        # 如果启用了自动优化
        if auto_optimize:
            print_info("开始自动优化...")
            _perform_auto_optimization(
                model=model,
                model_type=model_type,
                metrics=metrics,
                config=config,
                model_factory=model_factory
            )
        
    except Exception as e:
        print_error(f"评估失败: {str(e)}")
        raise typer.Exit(code=1)

@train_app.command(name="optimize", help="基于评估结果优化模型和特征工程")
def optimize_model(
    model_type: str = typer.Option(
        "xgboost", "--model", "-m",
        help="模型类型 (xgboost, random_forest, lstm)"
    ),
    evaluation_report: Optional[str] = typer.Option(
        None, "--report", "-r",
        help="评估报告路径，不指定则使用最新报告"
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
    """基于评估结果优化模型和特征工程"""
    # 配置日志
    setup_logging(log_level=log_level)
    
    # 加载配置
    try:
        config = ConfigManager(config_dir=config_dir)
    except Exception as e:
        print_error(f"加载配置失败: {str(e)}")
        raise typer.Exit(code=1)
    
    # 初始化反馈优化器
    optimizer = FeedbackOptimizer(config=config)
    
    # 获取评估指标
    metrics = {}
    feature_importance = None
    
    if evaluation_report:
        # 从指定报告加载评估结果
        try:
            import json
            with open(evaluation_report, 'r') as f:
                report_data = json.load(f)
            metrics = report_data.get("overall_metrics", {})
            feature_importance = report_data.get("feature_importance")
            print_info(f"已从 {evaluation_report} 加载评估结果")
        except Exception as e:
            print_error(f"加载评估报告失败: {str(e)}")
            raise typer.Exit(code=1)
    else:
        # 需要用户提供评估指标
        print_error("请提供评估报告路径或手动输入评估指标")
        raise typer.Exit(code=1)
    
    # 执行优化
    try:
        optimization_result = optimizer.optimize_based_on_evaluation(
            model_type=model_type,
            evaluation_metrics=metrics,
            feature_importance=feature_importance
        )
        
        print_success("优化完成!")
        if optimization_result.get("recommendations"):
            print_info("优化建议:")
            for i, recommendation in enumerate(optimization_result["recommendations"], 1):
                print(f"  {i}. {recommendation}")
        
        # 保存优化历史
        optimizer.save_optimization_history()
        print_info("优化历史已保存")
        
    except Exception as e:
        print_error(f"优化失败: {str(e)}")
        raise typer.Exit(code=1)

def _perform_auto_optimization(model, model_type, metrics, config, model_factory):
    """执行自动优化的辅助函数"""
    try:
        optimizer = FeedbackOptimizer(config=config)
        
        # 获取特征重要性
        feature_importance = None
        if hasattr(model, "get_feature_importance"):
            try:
                feature_importance = model.get_feature_importance()
            except Exception:
                pass
        
        # 执行优化
        optimization_result = optimizer.optimize_based_on_evaluation(
            model_type=model_type,
            evaluation_metrics=metrics,
            feature_importance=feature_importance,
            model_factory=model_factory
        )
        
        if optimization_result.get("recommendations"):
            print_info("自动优化建议:")
            for i, recommendation in enumerate(optimization_result["recommendations"], 1):
                print(f"  {i}. {recommendation}")
        
        # 保存优化历史
        optimizer.save_optimization_history()
        print_info("优化历史已保存")
        
    except Exception as e:
        print_warning(f"自动优化失败: {str(e)}")

def train_automl(
    data_path: Optional[str] = typer.Option(
        None, "--data", "-d",
        help="训练数据路径，不指定则使用默认数据目录"
    ),
    model_type: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="初始模型类型，不指定则自动选择"
    ),
    protocol: Optional[int] = typer.Option(
        None, "--protocol", "-p",
        help="协议类型（如6表示TCP）"
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
    """启动自动化机器学习训练，实现完整的训练-评估-优化-再训练闭环"""
    # 初始化日志
    setup_logging(log_level=log_level)
    
    # 加载配置
    try:
        config = ConfigManager(config_dir=config_dir)
    except Exception as e:
        print_error(f"加载配置失败: {str(e)}")
        raise typer.Exit(code=1)
    
    # 处理后台运行
    if background:
        try:
            # 简单的后台运行实现
            pid = os.fork()
            if pid > 0:
                print_success(f"AutoML训练已在后台启动 (PID: {pid})")
                raise typer.Exit(code=0)
        except OSError as e:
            print_error(f"无法在后台运行: {str(e)}")
            raise typer.Exit(code=1)
    
    # 初始化AutoML训练器
    trainer = AutoMLTrainer(config=config)
    
    # 启动AutoML训练
    print_info("启动AutoML训练过程...")
    try:
        success = trainer.start_automl_process(
            data_path=data_path,
            protocol=protocol,
            target_model_type=model_type,
            background=False  # 我们已经在当前线程中处理
        )
        
        if success:
            print_success("AutoML训练已完成")
            
            # 显示最佳模型信息
            best_model = trainer.best_model_info
            if best_model:
                print_info("最佳模型信息:")
                print(f"  迭代轮次: {best_model['iteration']}")
                print(f"  模型类型: {best_model['model_type']}")
                print(f"  F1分数: {best_model['f1_score']:.4f}")
                if best_model['metrics']:
                    print("  详细指标:")
                    for metric, value in best_model['metrics'].items():
                        print(f"    {metric}: {value:.4f}")
        else:
            print_error("启动AutoML训练失败")
            raise typer.Exit(code=1)
            
    except Exception as e:
        print_error(f"AutoML训练失败: {str(e)}")
        raise typer.Exit(code=1)

train_app.add_command(train_automl, name="automl")

def main():
    """模型训练相关操作"""
    train_app()

if __name__ == "__main__":
    main()