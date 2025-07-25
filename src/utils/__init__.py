"""
通用工具模块，提供日志、评估指标、数据处理和可视化等功能
"""

from .logger import get_logger, init_logger, set_log_level
from .metrics import (
    calculate_precision,
    calculate_recall,
    calculate_f1_score,
    calculate_auc,
    calculate_confusion_matrix,
    classification_report
)
from .helpers import (
    standardize_data,
    normalize_data,
    balance_dataset,
    train_test_split,
    save_json,
    load_json,
    save_pickle,
    load_pickle,
    timestamp_to_str,
    str_to_timestamp,
    get_ip_addresses,
    is_valid_ip,
    parse_packet_timestamp
)
from .evaluation_visualizer import EvaluationVisualizer

__all__ = [
    # 日志相关
    "get_logger",
    "init_logger",
    "set_log_level",
    # 评估指标
    "calculate_precision",
    "calculate_recall",
    "calculate_f1_score",
    "calculate_auc",
    "calculate_confusion_matrix",
    "classification_report",
    # 辅助函数
    "standardize_data",
    "normalize_data",
    "balance_dataset",
    "train_test_split",
    "save_json",
    "load_json",
    "save_pickle",
    "load_pickle",
    "timestamp_to_str",
    "str_to_timestamp",
    "get_ip_addresses",
    "is_valid_ip",
    "parse_packet_timestamp",
    # 可视化
    "EvaluationVisualizer"
]

__version__ = "1.0.0"
