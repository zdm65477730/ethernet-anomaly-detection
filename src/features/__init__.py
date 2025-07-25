"""
特征提取模块，负责从网络流量中提取各类特征

包括统计特征、时序特征，并支持不同协议的专属特征提取
"""

from .base_extractor import BaseFeatureExtractor
from .stat_extractor import StatFeatureExtractor
from .temporal_extractor import TemporalFeatureExtractor
from .protocol_specs import (
    get_protocol_spec,
    get_protocol_number,
    get_all_protocols,
    get_protocol_key_features,
    get_protocol_model_preference,
    is_feature_relevant
)

__all__ = [
    "BaseFeatureExtractor",
    "StatFeatureExtractor",
    "TemporalFeatureExtractor",
    "get_protocol_spec",
    "get_protocol_number",
    "get_all_protocols",
    "get_protocol_key_features",
    "get_protocol_model_preference",
    "is_feature_relevant"
]
__version__ = "1.0.0"
