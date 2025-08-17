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
            'external/local_xla', 'eigen', 'registering factory',
            'attempting to register', 'please check linkage'
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
from typing import Optional
import typer
import numpy as np
import pandas as pd
from src.data.data_processor import DataProcessor
from src.models.model_factory import ModelFactory
from src.models.model_selector import ModelSelector
from src.detection.anomaly_detector import AnomalyDetector
from src.config.config_manager import ConfigManager
from src.cli.utils import print_info, print_success, print_error, print_warning
from src.utils.logger import setup_logging, get_logger

app = typer.Typer(help="离线检测命令")

@app.command()
def file(
    data_path: str = typer.Option(
        ..., "--data-path", "-d",
        help="输入数据文件路径（CSV格式）"
    ),
    model_path: Optional[str] = typer.Option(
        None, "--model-path", "-m",
        help="模型文件路径（如果不指定，将自动选择最佳模型）"
    ),
    protocol: Optional[int] = typer.Option(
        None, "--protocol", "-p",
        help="协议类型（如果不指定，将尝试自动检测）"
    ),
    output_path: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="输出结果文件路径（JSON格式）"
    ),
    config_dir: str = typer.Option(
        "config", "--config-dir", "-c",
        help="配置文件目录"
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", "-l",
        help="日志级别 (DEBUG, INFO, WARNING, ERROR)"
    ),
    parallel: bool = typer.Option(
        False, "--parallel", "-P",
        help="启用并行处理以提高效率"
    ),
    batch_size: int = typer.Option(
        1000, "--batch-size", "-b",
        help="并行处理的批次大小"
    ),
    workers: int = typer.Option(
        0, "--workers", "-w",
        help="并行工作进程数（0表示使用CPU核心数）"
    )
):
    """
    对本地数据文件进行离线异常检测
    
    示例:
    anomaly-detector detect file -d data/test_data.csv -p 6
    """
    _file_command(
        data_path=data_path,
        model_path=model_path,
        protocol=protocol,
        output_path=output_path,
        config_dir=config_dir,
        log_level=log_level,
        parallel=parallel,
        batch_size=batch_size,
        workers=workers
    )

def _make_json_serializable(obj):
    """将对象转换为JSON可序列化的格式"""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.Series):
        return _make_json_serializable(obj.to_dict())
    elif isinstance(obj, pd.DataFrame):
        return _make_json_serializable(obj.to_dict('records'))
    else:
        return str(obj)

def _file_command(
    data_path: str,
    model_path: Optional[str],
    protocol: Optional[int],
    output_path: Optional[str],
    config_dir: str,
    log_level: str,
    parallel: bool,
    batch_size: int,
    workers: int
):
    """
    对本地数据文件进行离线异常检测
    
    示例:
    anomaly-detector detect file -d data/test_data.csv -p 6
    """
    # 初始化日志
    log_level_mapping = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40
    }
    setup_logging(log_level=log_level_mapping.get(log_level.upper(), 20))
    logger = get_logger("cli.detect")
    
    print_info(f"开始离线异常检测: {data_path}")
    
    try:
        # 加载配置
        config = ConfigManager(config_dir=config_dir)
        
        # 检查输入文件是否存在
        if not os.path.exists(data_path):
            print_error(f"输入文件不存在: {data_path}")
            raise typer.Exit(code=1)
        
        # 加载数据
        print_info("加载数据文件...")
        df = pd.read_csv(data_path)
        print_success(f"成功加载 {len(df)} 条记录")
        
        # 初始化数据处理器
        data_processor = DataProcessor(config=config)
        
        # 预处理数据
        print_info("预处理数据...")
        try:
            processed_df = data_processor.preprocess_features(df, fit=False)
        except Exception as e:
            print_warning(f"数据预处理失败: {str(e)}，检查是否为模型兼容特征")
            # 检查是否已经是模型兼容特征
            model_compatible_features, _ = data_processor.get_model_compatible_features()
            available_features = [f for f in model_compatible_features if f in df.columns]
            
            if len(available_features) > 0:
                processed_df = df[available_features]
                print_info(f"使用 {len(available_features)} 个模型兼容特征")
            else:
                print_warning("未找到模型兼容特征，使用原始数据")
                processed_df = df
        
        # 如果没有指定协议，尝试从数据中推断
        if protocol is None and 'protocol' in df.columns:
            # 尝试使用最常见的协议
            protocol = int(df['protocol'].mode().iloc[0]) if not df['protocol'].mode().empty else 6
            print_info(f"自动检测到协议类型: {protocol}")
        
        # 初始化模型工厂和选择器
        model_factory = ModelFactory(config=config)
        model_selector = ModelSelector(config=config)
        
        # 加载模型
        model = None
        if model_path:
            # 从指定路径加载模型
            # 检查模型文件是否存在（支持相对路径和绝对路径）
            model_file_path = model_path
            if not os.path.exists(model_file_path):
                # 尝试在当前目录下查找
                model_file_path = os.path.join(os.getcwd(), model_path)
                if not os.path.exists(model_file_path):
                    # 尝试在模型目录下查找
                    models_dir = config.get("model.models_dir", "models")
                    model_file_path = os.path.join(models_dir, model_path)
                    if not os.path.exists(model_file_path):
                        print_error(f"模型文件不存在: {model_path}")
                        raise typer.Exit(code=1)
            
            try:
                # 从模型文件名推断模型类型
                model_filename = os.path.basename(model_file_path)
                if 'xgboost' in model_filename:
                    model_type = 'xgboost'
                elif 'random_forest' in model_filename:
                    model_type = 'random_forest'
                elif 'lstm' in model_filename:
                    model_type = 'lstm'
                elif 'mlp' in model_filename:
                    model_type = 'mlp'
                elif 'ensemble' in model_filename:
                    model_type = 'ensemble'
                else:
                    # 默认使用配置中的默认模型类型
                    model_type = config.get("model.default_type", "xgboost")
                    print_warning(f"无法从文件名推断模型类型，使用默认模型类型: {model_type}")
                
                # 加载模型
                if model_type == 'ensemble':
                    from src.models.ensemble_model import EnsembleModel
                    model = EnsembleModel.load(model_file_path)
                else:
                    model = model_factory.load_model(model_type, model_file_path)
                    
                print_success(f"成功加载模型: {model_file_path}")
            except Exception as e:
                print_error(f"加载模型失败: {str(e)}")
                raise typer.Exit(code=1)
        else:
            # 自动选择模型
            if protocol is not None:
                model_type = model_selector.select_best_model(protocol)
                model = model_factory.load_latest_model(model_type)
                if model:
                    print_info(f"为协议 {protocol} 自动选择模型: {model_type}")
                else:
                    print_warning(f"未找到协议 {protocol} 的模型，将使用规则检测")
            else:
                # 使用默认模型
                model_type = config.get("model.default_type", "xgboost")
                model = model_factory.load_latest_model(model_type)
                if model:
                    print_info(f"使用默认模型: {model_type}")
                else:
                    print_warning(f"未找到默认模型，将使用规则检测")
        
        # 初始化异常检测器
        detector = AnomalyDetector(config=config)
        
        # 如果加载了模型，将其设置到检测器中
        if model and protocol is not None:
            detector.models[protocol] = model
        elif model:
            # 如果没有指定协议，使用默认协议0
            detector.models[0] = model
        
        # 执行检测
        print_info("开始执行异常检测...")
        detection_results = []
        
        # 如果启用并行处理且数据量较大
        if parallel and len(processed_df) > batch_size:
            print_info("使用并行处理进行异常检测...")
            
            # 确定工作进程数
            import multiprocessing
            if workers == 0:
                workers = multiprocessing.cpu_count()
            
            # 分批处理数据
            from multiprocessing import Pool
            batches = [processed_df[i:i+batch_size] for i in range(0, len(processed_df), batch_size)]
            
            # 准备进程池参数
            pool_args = [(batch, model_path, protocol, config_dir) for batch in batches]
            
            try:
                with Pool(processes=workers) as pool:
                    batch_results = pool.map(_detect_batch, pool_args)
                    
                # 合并结果
                for batch_result in batch_results:
                    detection_results.extend(batch_result)
                    
                print_success(f"并行处理完成，检测到 {len([r for r in detection_results if r.get('is_anomaly', False)])} 个异常")
            except Exception as e:
                print_error(f"并行处理失败: {str(e)}，回退到单线程处理")
                # 回退到单线程处理
                _perform_sequential_detection(detector, processed_df, detection_results)
        else:
            # 单线程处理
            _perform_sequential_detection(detector, processed_df, detection_results)
        
        # 保存结果
        if output_path:
            output_file = output_path
        else:
            # 默认输出路径
            reports_dir = config.get("detection.reports_dir", "reports/detections")
            os.makedirs(reports_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(reports_dir, f"detection_results_{timestamp}.json")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 确保所有数据都是JSON可序列化的
        detection_results = _make_json_serializable(detection_results)
        
        # 保存结果到文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detection_results, f, ensure_ascii=False, indent=2)
        
        print_success(f"检测完成，结果已保存到: {output_file}")
        print_info(f"总共检测到 {len([r for r in detection_results if r.get('is_anomaly', False)])} 个异常")
        
    except Exception as e:
        print_error(f"离线异常检测过程中出错: {str(e)}")
        logger.error("离线异常检测过程中出错", exc_info=True)
        raise typer.Exit(code=1)

def _perform_sequential_detection(detector, processed_df, detection_results):
    """执行顺序检测"""
    total_records = len(processed_df)
    print_info(f"使用单线程处理 {total_records} 条记录...")
    
    for idx, (_, row) in enumerate(processed_df.iterrows()):
        try:
            features = row.to_dict()
            result = detector.detect_anomaly(features)
            if result:
                detection_results.append(result)
                
            # 显示进度
            if (idx + 1) % 1000 == 0 or (idx + 1) == total_records:
                print_info(f"已处理 {idx + 1}/{total_records} 条记录")
                
        except Exception as e:
            print_warning(f"处理第 {idx + 1} 条记录时出错: {str(e)}")

def _detect_batch(args):
    """全局批处理函数，用于并行异常检测"""
    batch_data, model_path, protocol, config_dir = args
    from src.detection.anomaly_detector import AnomalyDetector
    from src.models.model_factory import ModelFactory
    from src.config.config_manager import ConfigManager
    
    # 重新初始化配置和模型工厂
    config_manager = ConfigManager(config_dir)
    model_factory = ModelFactory(config_manager)
    
    # 初始化异常检测器
    detector = AnomalyDetector(config=config_manager)
    
    # 加载模型
    if model_path:
        try:
            # 从模型文件名推断模型类型
            model_filename = os.path.basename(model_path)
            if 'xgboost' in model_filename:
                model_type = 'xgboost'
            elif 'random_forest' in model_filename:
                model_type = 'random_forest'
            elif 'lstm' in model_filename:
                model_type = 'lstm'
            elif 'mlp' in model_filename:
                model_type = 'mlp'
            elif 'ensemble' in model_filename:
                model_type = 'ensemble'
            else:
                # 默认使用xgboost
                model_type = 'xgboost'
            
            # 加载模型
            if model_type == 'ensemble':
                from src.models.ensemble_model import EnsembleModel
                model = EnsembleModel.load(model_path)
            else:
                model = model_factory.load_model(model_type, model_path)
            
            # 设置模型到检测器中
            detector.models[protocol or 0] = model
        except Exception as e:
            print(f"并行进程中加载模型失败: {str(e)}")
            return []
    
    results = []
    for _, row in batch_data.iterrows():
        features = row.to_dict()
        result = detector.detect_anomaly(features)
        if result:
            results.append(result)
    return results

if __name__ == "__main__":
    app()