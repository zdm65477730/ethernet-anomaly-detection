import os
import sys

# 在绝对最早期设置环境变量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 在绝对最早期重定向stderr
class DevNull:
    def write(self, msg):
        # 检查是否包含TensorFlow相关关键词
        tf_keywords = [
            'tensorflow', 'cuda', 'cudnn', 'cufft', 'cublas', 
            'absl', 'computation_placer', 'oneDNN', 'stream_executor',
            'external/local_xla'
        ]
        
        msg_lower = str(msg).lower()
        # 如果包含任何TF关键词，直接丢弃
        for keyword in tf_keywords:
            if keyword in msg_lower:
                return
        # 否则正常输出到原始stderr
        sys.__stderr__.write(msg)
    
    def flush(self):
        sys.__stderr__.flush()

# 重定向stderr到我们的过滤器
sys.stderr = DevNull()

# 导入其他模块
import json
import time
import logging
import signal
from typing import Optional
import typer
import numpy as np
import pandas as pd

# 设置Python日志级别
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

from src.data.data_processor import DataProcessor
from src.data.data_storage import DataStorage
from src.models.model_factory import ModelFactory
from src.models.model_selector import ModelSelector
from src.training.model_trainer import ModelTrainer
from src.training.model_evaluator import ModelEvaluator
from src.training.feedback_optimizer import FeedbackOptimizer
from src.training.continuous_trainer import ContinuousTrainer
from src.detection.feedback_processor import FeedbackProcessor
from src.training.automl_trainer import AutoMLTrainer
from src.config.config_manager import ConfigManager
from src.utils.logger import setup_logging
from src.cli.utils import (
    print_info,
    print_error,
    print_success,
    print_warning,
    confirm
)

# 创建子命令
train_app = typer.Typer(help="模型训练相关命令", invoke_without_command=True)


@train_app.callback()
def main(
    ctx: typer.Context,
    model_type: str = typer.Option(
        "xgboost", "--model", "-m",
        help="模型类型 (xgboost, random_forest, lstm)"
    ),
    data_path: Optional[str] = typer.Option(
        None, "--data", "-d",
        help="训练数据路径，不指定则使用默认数据目录"
    ),
    test_data: Optional[str] = typer.Option(
        None, "--test-data", "-td",
        help="测试数据路径，不指定则使用默认测试数据目录"
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
    # 如果是直接调用命令而不是子命令，则执行训练
    if ctx.invoked_subcommand is None:
        # 配置日志
        log_level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR
        }
        actual_log_level = log_level_map.get(log_level.upper(), logging.INFO)
        setup_logging(log_level=actual_log_level)
        
        # 加载配置
        try:
            config = ConfigManager(config_dir=config_dir)
        except Exception as e:
            print_error(f"加载配置失败: {str(e)}")
            raise typer.Exit(code=1)
        
        # 确定交叉验证折数
        if cv_folds is None:
            cv_folds = config.get("training.cross_validation_folds", 5)
        else:
            cv_folds = int(cv_folds)  # 确保是整数类型
        
        # 初始化组件
        data_processor = DataProcessor()
        model_factory = ModelFactory()
        
        # 确定数据路径
        if not data_path:
            data_path = os.path.join(config.get("data.processed_dir", "data/processed"))
        
        # 确定测试数据路径
        if not test_data:
            test_dir = config.get("data.test_dir", "data/test")
        else:
            test_dir = test_data
        
        # 确保路径不为None
        if data_path is None:
            data_path = "data/processed"
            
        if test_dir is None:
            test_dir = "data/test"
        
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
            
            # 预处理训练数据
            # 只使用模型兼容的特征
            model_compatible_features, _ = data_processor.get_model_compatible_features()
            available_features = [f for f in model_compatible_features if f in X.columns]
            X_filtered = X[available_features]
            X_processed = data_processor.preprocess_features(X_filtered, fit=True)
            
            # 更新X和y为处理后的数据
            X = X_processed
            
            # 如果没有测试数据，使用训练数据的一部分
            if not os.path.exists(test_dir) or len(os.listdir(test_dir)) == 0:
                print_warning(f"测试数据目录 {test_dir} 不存在或为空，使用训练数据的一部分作为测试集")
                _, X_test, _, y_test = data_processor.split_train_test(X, y, test_size=0.2)
                print_info(f"已从训练数据拆分出 {len(X_test)} 个测试样本")
                
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
                cv_folds=cv_folds or 5,  # 确保cv_folds是一个整数，默认值为5
                output_dir=output_dir
            )
            
            
            print_success("模型训练完成!")
            print_info("评估指标:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
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

def _create_continuous_training_pid_file(pid: int):
    """创建持续训练PID文件"""
    pid_file = "continuous_training.pid"
    try:
        with open(pid_file, "w") as f:
            f.write(str(pid))
    except Exception as e:
        print_warning(f"创建持续训练PID文件失败: {e}")

def _remove_continuous_training_pid_file():
    """删除持续训练PID文件"""
    pid_file = "continuous_training.pid"
    try:
        if os.path.exists(pid_file):
            os.remove(pid_file)
    except Exception as e:
        print_warning(f"删除持续训练PID文件失败: {e}")

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
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }
    actual_log_level = log_level_map.get(log_level.upper(), logging.INFO)
    setup_logging(log_level=actual_log_level)
    
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
                # 创建PID文件记录后台进程ID
                _create_continuous_training_pid_file(pid)
                raise typer.Exit(code=0)
            else:
                # 子进程中设置新的进程组，以便可以独立接收信号
                os.setsid()
        except OSError as e:
            print_error(f"无法在后台运行: {str(e)}")
            raise typer.Exit(code=1)
    
    # 如果不是后台运行，也创建PID文件
    if not background:
        _create_continuous_training_pid_file(os.getpid())
    
    # 初始化持续训练器
    trainer = ContinuousTrainer(config=config)
    
    # 启动持续训练
    print_info(f"启动持续训练模式，检查间隔: {check_interval}秒，最小样本数: {min_samples}")
    try:
        trainer.start(
            check_interval=check_interval,
            min_samples=min_samples
        )
        
        # 注册信号处理器以便优雅停止
        def signal_handler(sig, frame):
            print_info("\n收到停止信号，正在停止持续训练...")
            trainer.stop()
            _remove_continuous_training_pid_file()
            print_success("持续训练已停止")
            os._exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # 保持运行
        try:
            while True:
                time.sleep(1)  # 更短的睡眠时间，以便更快响应信号
        except KeyboardInterrupt:
            print_info("\n用户中断，正在停止持续训练...")
            trainer.stop()
        
        _remove_continuous_training_pid_file()
        print_success("持续训练已停止")
        
    except Exception as e:
        print_error(f"持续训练失败: {str(e)}")
        trainer.stop()
        _remove_continuous_training_pid_file()
        raise typer.Exit(code=1)

@train_app.command(name="evaluate", help="评估现有模型")
def evaluate_model(
    ctx: typer.Context,
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
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", "-l",
        help="日志级别 (DEBUG, INFO, WARNING, ERROR)"
    )
):
    """评估已有模型"""
    # 如果是直接调用命令而不是子命令，则执行评估
    if ctx.invoked_subcommand is None:
        # 配置日志
        log_level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR
        }
        actual_log_level = log_level_map.get(log_level.upper(), logging.INFO)
        setup_logging(log_level=actual_log_level)
        
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
                X_test, y_test = None, None
                if os.path.exists(test_dir):
                    try:
                        X_test, y_test = data_processor.load_processed_data(test_dir)
                    except Exception as e:
                        print_warning(f"加载测试数据失败: {e}")
                
                # 如果没有测试数据或加载失败，使用训练数据的一部分
                if X_test is None or len(X_test) == 0:
                    print_warning(f"测试数据目录 {test_dir} 不存在或为空，使用训练数据的一部分作为测试集")
                    train_dir = config.get("data.processed_dir", "data/processed")
                    X, y = data_processor.load_processed_data(train_dir)
                    _, X_test, _, y_test = data_processor.split_train_test(X, y, test_size=0.2)
            
            if X_test is None or len(X_test) == 0:
                print_error("没有可用的测试数据")
                raise typer.Exit(code=1)
                
            # 对测试数据进行预处理
            print_info(f"原始测试数据形状: {X_test.shape}")
            # 先清理数据
            X_test_cleaned = data_processor.clean_data(X_test)
            print_info(f"清理后的测试数据形状: {X_test_cleaned.shape}")
            
            # 获取模型兼容的特征集
            model_compatible_features, _ = data_processor.get_model_compatible_features()
            available_features = [f for f in model_compatible_features if f in X_test_cleaned.columns]
            
            if len(available_features) != len(model_compatible_features):
                print_warning(f"特征不完整，期望: {model_compatible_features}, 可用: {available_features}")
            
            # 只保留模型兼容的特征
            X_test_filtered = X_test_cleaned[available_features]
            X_test = data_processor.preprocess_features(X_test_filtered, fit=True)
            print_info(f"预处理后的测试数据形状: {X_test.shape}")
            
            # 检查特征数量是否匹配
            if hasattr(model, 'model') and hasattr(model.model, 'n_features_in_'):
                expected_features_count = model.model.n_features_in_
                actual_features_count = X_test.shape[1] if hasattr(X_test, 'shape') else len(X_test)
                if actual_features_count != expected_features_count:
                    print_warning(f"特征数量不匹配，模型期望: {expected_features_count}, 实际: {actual_features_count}")
                    # 尝试调整特征数量
                    if actual_features_count > expected_features_count:
                        print_info(f"截取前{expected_features_count}个特征")
                        if hasattr(X_test, 'iloc'):
                            X_test = X_test.iloc[:, :expected_features_count]
                        else:
                            X_test = X_test[:, :expected_features_count]
                    else:
                        print_error(f"特征数量不足，无法进行评估")
                        raise typer.Exit(code=1)
            
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
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
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
    feedback_based: bool = typer.Option(
        False, "--feedback-based", "-f",
        help="基于反馈数据进行优化"
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
    
    if feedback_based:
        # 基于反馈数据进行优化
        try:
            feedback_processor = FeedbackProcessor(config=config)
            feedback_data = feedback_processor.get_feedback_data()
            
            # 从反馈数据中提取评估指标
            metrics = feedback_processor.calculate_metrics_from_feedback()
            print_info("已从反馈数据加载评估指标")
        except Exception as e:
            print_error(f"加载反馈数据失败: {str(e)}")
            raise typer.Exit(code=1)
    elif evaluation_report:
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
        print_error("请提供评估报告路径或使用--feedback-based选项")
        raise typer.Exit(code=1)
    
    # 执行优化
    try:
        optimization_result = optimizer.optimize_based_on_evaluation(
            model_type=model_type,
            evaluation_metrics=metrics,
            feature_importance=feature_importance
        )
        
        # 输出优化结果
        print_success("优化完成!")
        import json
        print(json.dumps(optimization_result, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print_error(f"优化过程中发生错误: {str(e)}")
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

# 注册AutoML训练命令
train_app.command(name="automl", help="启动AutoML自动化模型训练")(train_automl)

def main():
    """模型训练相关操作"""
    train_app()

if __name__ == "__main__":
    main()