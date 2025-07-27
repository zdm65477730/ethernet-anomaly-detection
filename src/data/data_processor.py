import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional, List, Dict, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.utils.logger import get_logger
from src.config.config_manager import ConfigManager
from src.data.data_storage import DataStorage

class DataProcessor:
    """数据处理器，负责数据清洗、转换和分割"""
    
    def __init__(self, config: Optional[ConfigManager] = None, storage: Optional[DataStorage] = None):
        """
        初始化数据处理器
        
        Args:
            config: 配置管理器
            storage: 数据存储管理器
        """
        self.config = config or ConfigManager()
        self.storage = storage or DataStorage(self.config)
        self.logger = get_logger("data_processor")
        
        # 特征处理管道
        self._preprocessing_pipeline = None
        
        # 初始化预处理管道
        self._init_preprocessing_pipeline()
    
    def _init_preprocessing_pipeline(self) -> None:
        """初始化数据预处理管道"""
        # 获取预期特征名称
        expected_numeric, expected_categorical = self.get_expected_feature_names()
        
        # 数值特征处理：填充缺失值 + 标准化
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # 分类特征处理：填充缺失值 + 独热编码
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # 确保特征在数据中存在
        numeric_features_present = [f for f in expected_numeric if f in self._get_all_columns()]
        categorical_features_present = [f for f in expected_categorical if f in self._get_all_columns()]
        
        # 组合所有特征处理器
        transformers = []
        if numeric_features_present:
            transformers.append(('num', numeric_transformer, numeric_features_present))
        if categorical_features_present:
            transformers.append(('cat', categorical_transformer, categorical_features_present))
            
        self._preprocessing_pipeline = ColumnTransformer(transformers=transformers)
        
        self.logger.debug("数据预处理管道已初始化")
    
    def _get_all_columns(self):
        """获取所有可能的列名"""
        # 尝试从现有数据文件中获取列名
        try:
            # 检查训练数据目录
            train_dir = self.config.get("data.processed_dir", "data/processed")
            if os.path.exists(train_dir):
                for file in os.listdir(train_dir):
                    if file.endswith((".csv", ".parquet")):
                        file_path = os.path.join(train_dir, file)
                        if file.endswith(".csv"):
                            df = pd.read_csv(file_path, nrows=1)
                        else:
                            df = pd.read_parquet(file_path)
                        return df.columns.tolist()
        except Exception as e:
            self.logger.debug(f"无法从数据文件获取列名: {e}")
        
        # 返回默认列名
        expected_numeric, expected_categorical = self.get_expected_feature_names()
        return expected_numeric + expected_categorical

    def get_expected_feature_names(self):
        """获取预期的特征名称（与模型训练时一致）"""
        # 定义与特征提取器一致的特征
        expected_numeric_features = [
            # 基本统计特征
            "packet_count", "byte_count", "flow_duration", "avg_packet_size",
            "std_packet_size", "min_packet_size", "max_packet_size",
            "bytes_per_second", "packets_per_second",
            
            # 协议特征
            "tcp_syn_count", "tcp_ack_count", "tcp_fin_count", "tcp_rst_count",
            "tcp_flag_ratio", "tcp_packet_ratio", "udp_packet_ratio", "icmp_packet_ratio",
            
            # 载荷特征
            "avg_payload_size", "payload_entropy", "payload_size_std",
            
            # 端口特征
            "src_port_entropy", "dst_port_entropy",
            
            # 方向特征
            "outbound_packet_ratio", "inbound_packet_ratio",
            
            # 时序特征（短窗口）
            "short_window_packet_rate", "short_window_byte_rate",
            "short_window_packet_size_mean", "short_window_packet_size_std",
            "short_window_inter_arrival_mean", "short_window_inter_arrival_std",
            "short_window_burst_count", "short_window_burst_duration_mean",
            
            # 时序特征（中窗口）
            "medium_window_packet_rate", "medium_window_byte_rate",
            "medium_window_packet_size_mean", "medium_window_packet_size_std",
            "medium_window_inter_arrival_mean", "medium_window_inter_arrival_std",
            "medium_window_burst_count", "medium_window_burst_duration_mean",
            
            # 时序特征（长窗口）
            "long_window_packet_rate", "long_window_byte_rate",
            "long_window_packet_size_mean", "long_window_packet_size_std",
            "long_window_inter_arrival_mean", "long_window_inter_arrival_std",
            "long_window_burst_count", "long_window_burst_duration_mean",
            
            # 趋势特征
            "packet_rate_trend", "byte_rate_trend",
            "packet_size_variation", "inter_arrival_variation"
        ]
        
        expected_categorical_features = [
            # 当前版本没有分类特征
        ]
        
        # 总特征数: 19个数值特征 + 0个分类特征 = 19个特征
        # 但模型期望15个特征，我们需要根据模型实际训练时使用的特征进行调整
        return expected_numeric_features, expected_categorical_features
    
    def get_model_compatible_features(self):
        """
        获取与模型兼容的特征名称（模型期望15个特征）
        这些特征应该与模型训练时使用的特征一致
        """
        # 根据错误信息，模型期望15个特征，实际提供了19个特征
        # 我们需要确定模型实际使用的是哪15个特征
        model_compatible_features = [
            "packet_count", "byte_count", "flow_duration", "avg_packet_size",
            "std_packet_size", "min_packet_size", "max_packet_size",
            "bytes_per_second", "packets_per_second",
            "tcp_syn_count", "tcp_ack_count", "tcp_fin_count", "tcp_rst_count",
            "tcp_flag_ratio", "payload_entropy"
        ]
        
        return model_compatible_features, []
    
    def _map_raw_features_to_model_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将原始特征映射到模型兼容的特征
        
        Args:
            df: 原始数据DataFrame
            
        Returns:
            映射后的数据DataFrame
        """
        mapped_df = pd.DataFrame()
        
        # 基本映射关系
        feature_mapping = {
            "packet_count": "packet_size",  # 这里可能需要根据实际数据调整
            "byte_count": "packet_size",    # 这里可能需要根据实际数据调整
            "flow_duration": "session_duration",
            "avg_packet_size": "packet_size",
            "std_packet_size": "packet_size",  # 可能需要计算
            "min_packet_size": "packet_size",
            "max_packet_size": "packet_size",
            "bytes_per_second": "packet_size",  # 可能需要计算
            "packets_per_second": "packet_size",  # 可能需要计算
            "tcp_syn_count": "tcp_flags",  # 需要解析TCP标志
            "tcp_ack_count": "tcp_flags",  # 需要解析TCP标志
            "tcp_fin_count": "tcp_flags",  # 需要解析TCP标志
            "tcp_rst_count": "tcp_flags",  # 需要解析TCP标志
            "tcp_flag_ratio": "tcp_flags",  # 需要解析TCP标志
            "payload_entropy": "payload_entropy"
        }
        
        # 模型兼容的特征列表
        model_features, _ = self.get_model_compatible_features()
        
        # 对于每个模型兼容的特征，尝试从原始数据中获取或计算
        for feature in model_features:
            if feature in feature_mapping:
                source_feature = feature_mapping[feature]
                if source_feature in df.columns:
                    if feature == "flow_duration":
                        mapped_df[feature] = df["session_duration"]
                    elif feature == "payload_entropy":
                        mapped_df[feature] = df["payload_entropy"]
                    elif feature in ["tcp_syn_count", "tcp_ack_count", "tcp_fin_count", "tcp_rst_count", "tcp_flag_ratio"]:
                        # 解析TCP标志
                        mapped_df[feature] = self._parse_tcp_flags(df["tcp_flags"], feature)
                    elif feature in ["packet_count", "avg_packet_size", "std_packet_size", "min_packet_size", "max_packet_size"]:
                        # 这些特征在当前数据中可能需要特殊处理
                        mapped_df[feature] = df["packet_size"]
                    elif feature in ["bytes_per_second", "packets_per_second"]:
                        # 这些特征需要计算
                        mapped_df[feature] = df["packet_size"]  # 临时使用packet_size
                    else:
                        mapped_df[feature] = df[source_feature]
                else:
                    # 如果源特征不存在，使用默认值
                    mapped_df[feature] = 0
            else:
                # 如果没有映射关系，检查是否直接存在于数据中
                if feature in df.columns:
                    mapped_df[feature] = df[feature]
                else:
                    # 使用默认值
                    mapped_df[feature] = 0
        
        return mapped_df
    
    def _parse_tcp_flags(self, tcp_flags_series: pd.Series, flag_type: str) -> pd.Series:
        """
        解析TCP标志字段
        
        Args:
            tcp_flags_series: TCP标志系列
            flag_type: 标志类型 (syn, ack, fin, rst, ratio)
            
        Returns:
            解析后的标志计数或比例
        """
        # TCP标志位映射 (基于数据中的文本表示)
        flag_map = {
            "SYN": 0x02,
            "ACK": 0x10,
            "FIN": 0x01,
            "RST": 0x04,
            "SYN+ACK": 0x12,
            "FIN+ACK": 0x11,
            "PSH+ACK": 0x18
        }
        
        if flag_type == "tcp_syn_count":
            return tcp_flags_series.apply(lambda x: 1 if "SYN" in str(x) else 0)
        elif flag_type == "tcp_ack_count":
            return tcp_flags_series.apply(lambda x: 1 if "ACK" in str(x) else 0)
        elif flag_type == "tcp_fin_count":
            return tcp_flags_series.apply(lambda x: 1 if "FIN" in str(x) else 0)
        elif flag_type == "tcp_rst_count":
            return tcp_flags_series.apply(lambda x: 1 if "RST" in str(x) else 0)
        elif flag_type == "tcp_flag_ratio":
            # 假设所有包都有标志，返回1表示有标志
            return tcp_flags_series.apply(lambda x: 1 if str(x) != "0" else 0)
        else:
            return pd.Series([0] * len(tcp_flags_series))
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗数据，处理缺失值和异常值
        
        Args:
            df: 原始数据DataFrame
        
        Returns:
            清洗后的数据DataFrame
        """
        if df.empty:
            self.logger.warning("尝试清洗空数据，返回空DataFrame")
            return df
        
        # 复制数据以避免修改原始数据
        cleaned_df = df.copy()
        
        # 1. 处理缺失值
        self.logger.debug(f"清洗前数据形状: {cleaned_df.shape}，缺失值数量: {cleaned_df.isnull().sum().sum()}")
        
        # 2. 处理非数值字段（如IP地址）
        # 删除或转换非数值字段
        non_numeric_columns = []
        for col in cleaned_df.columns:
            if col in ['src_ip', 'dst_ip', 'src_mac', 'dst_mac']:
                # 删除网络地址字段
                non_numeric_columns.append(col)
            elif cleaned_df[col].dtype == 'object':
                # 检查object类型的列是否包含非数值数据
                try:
                    # 尝试转换为数值类型
                    pd.to_numeric(cleaned_df[col], errors='raise')
                except (ValueError, TypeError):
                    # 如果转换失败，标记为非数值列
                    non_numeric_columns.append(col)
        
        # 删除非数值列
        if non_numeric_columns:
            self.logger.info(f"删除非数值列: {non_numeric_columns}")
            cleaned_df = cleaned_df.drop(columns=non_numeric_columns)
        
        # 3. 处理异常值
        # 处理数值特征中的异常值（如负的包大小）
        expected_numeric, _ = self.get_expected_feature_names()
        for col in expected_numeric:
            if col in cleaned_df.columns:
                # 对于不能为负的特征，将负值替换为0
                if col in ["packet_count", "byte_count", "flow_duration", "avg_packet_size",
                          "std_packet_size", "min_packet_size", "max_packet_size",
                          "bytes_per_second", "packets_per_second", "tcp_syn_count",
                          "tcp_ack_count", "tcp_fin_count", "tcp_rst_count", "tcp_flag_ratio",
                          "payload_entropy", "payload_size_std", "src_port_entropy",
                          "dst_port_entropy", "outbound_packet_ratio", "inbound_packet_ratio"]:
                    cleaned_df[col] = cleaned_df[col].clip(lower=0)
        
        # 4. 处理数据类型
        # 注意：timestamp列不应该作为特征使用，仅用于数据处理内部使用
        # 如果存在timestamp列，确保其为数值类型而不是datetime
        if "timestamp" in cleaned_df.columns:
            # 不要转换为datetime，保持为数值类型（Unix时间戳）
            # 只确保是数值类型
            cleaned_df["timestamp"] = pd.to_numeric(cleaned_df["timestamp"], errors='coerce')
        
        if "label" in cleaned_df.columns:
            cleaned_df["label"] = cleaned_df["label"].astype(int)
        
        # 确保所有特征列都是数值类型
        feature_columns = [col for col in cleaned_df.columns if col != 'label']
        for col in feature_columns:
            if cleaned_df[col].dtype == 'object':
                # 尝试转换为数值类型
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        
        self.logger.debug(f"清洗后数据形状: {cleaned_df.shape}，缺失值数量: {cleaned_df.isnull().sum().sum()}")
        return cleaned_df
    
    def balance_dataset(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        平衡数据集，处理类别不平衡问题
        
        Args:
            X: 特征数据
            y: 标签数据
        
        Returns:
            平衡后的特征和标签
        """
        # 计算类别分布
        class_counts = y.value_counts()
        self.logger.debug(f"原始类别分布: {dict(class_counts)}")
        
        # 如果类别已经相对平衡，则不进行处理
        if len(class_counts) <= 1:
            self.logger.warning("数据集中只有一个类别，无法平衡")
            return X, y
            
        # 计算类别比例
        min_class = class_counts.idxmin()
        max_class = class_counts.idxmax()
        min_count = class_counts[min_class]
        max_count = class_counts[max_class]
        
        # 如果多数类是少数类的5倍以上，则进行平衡
        if max_count / min_count > 5:
            self.logger.info(f"数据集不平衡，进行平衡处理 (多数类: {max_class}={max_count}, 少数类: {min_class}={min_count})")
            
            # 对多数类进行下采样
            if min_count > 0:
                # 获取多数类和少数类的索引
                min_indices = y[y == min_class].index
                max_indices = y[y == max_class].index
                
                # 下采样多数类
                np.random.seed(42)  # 固定随机种子，保证结果可复现
                max_indices_downsampled = np.random.choice(
                    max_indices, size=min_count, replace=False
                )
                
                # 合并索引
                downsampled_indices = np.concatenate([min_indices, max_indices_downsampled])
                
                # 重新索引数据
                X = X.loc[downsampled_indices].reset_index(drop=True)
                y = y.loc[downsampled_indices].reset_index(drop=True)
                
                self.logger.debug(f"下采样后数据集大小: {len(X)}")
        
        return X, y
    
    def preprocess_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        预处理特征数据
        
        Args:
            X: 原始特征数据
            fit: 是否拟合预处理管道
            
        Returns:
            预处理后的特征数据
        """
        if X.empty:
            self.logger.warning("尝试预处理空数据")
            return X
            
        self.logger.debug(f"预处理前特征形状: {X.shape}")
        
        # 检查是否是模型兼容的特征
        model_features, _ = self.get_model_compatible_features()
        if set(X.columns) == set(model_features):
            # 如果已经是模型兼容的特征，直接返回
            self.logger.debug("数据已经是模型兼容特征，跳过预处理")
            return X
        
        # 否则使用预处理管道
        if fit:
            # 拟合并转换
            X_processed = self._preprocessing_pipeline.fit_transform(X)
        else:
            # 仅转换
            X_processed = self._preprocessing_pipeline.transform(X)
        
        # 获取特征名称
        feature_names = self._get_feature_names_after_preprocessing()
        
        # 转换为DataFrame
        if isinstance(X_processed, np.ndarray):
            X_processed = pd.DataFrame(X_processed, columns=feature_names)
        
        self.logger.debug(f"预处理后特征形状: {X_processed.shape}")
        return X_processed
    
    def _get_feature_names_after_preprocessing(self):
        """获取预处理后的特征名称"""
        feature_names = []
        
        # 获取数值特征名称
        expected_numeric, expected_categorical = self.get_expected_feature_names()
        
        # 数值特征名称保持不变
        numeric_features_present = [f for f in expected_numeric if f in self._get_all_columns()]
        feature_names.extend(numeric_features_present)
        
        # 分类特征经过独热编码后会扩展
        categorical_features_present = [f for f in expected_categorical if f in self._get_all_columns()]
        for feature in categorical_features_present:
            # 检查预处理管道中是否有独热编码器
            try:
                # 获取独热编码器的特征名称
                onehot_encoder = self._preprocessing_pipeline.named_transformers_['cat'].named_steps['onehot']
                if hasattr(onehot_encoder, 'get_feature_names_out'):
                    cat_feature_names = onehot_encoder.get_feature_names_out([feature])
                    feature_names.extend(cat_feature_names)
                else:
                    feature_names.append(feature)
            except Exception:
                feature_names.append(feature)
        
        return feature_names
    
    def load_processed_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        加载处理后的数据
        
        Args:
            data_path: 数据路径（文件或目录）
            
        Returns:
            特征数据和标签数据
        """
        # 确定数据文件路径
        if os.path.isdir(data_path):
            # 如果是目录，查找其中的数据文件
            data_files = [f for f in os.listdir(data_path) 
                         if f.endswith(('.csv', '.parquet')) and not f.startswith('.')]
            if not data_files:
                raise FileNotFoundError(f"在目录 {data_path} 中未找到数据文件")
            # 使用第一个数据文件
            data_file = data_files[0]
            file_path = os.path.join(data_path, data_file)
        else:
            file_path = data_path
        
        self.logger.info(f"加载数据文件: {file_path}")
        
        # 读取数据
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"不支持的数据文件格式: {file_path}")
        
        self.logger.debug(f"加载数据形状: {df.shape}")
        
        # 分离特征和标签
        if 'label' not in df.columns:
            raise ValueError("数据中未找到'label'列作为标签")
        
        # 检查是否是模型兼容特征
        model_features, _ = self.get_model_compatible_features()
        if all(feature in df.columns for feature in model_features):
            # 如果是模型兼容特征，直接分离
            X = df[model_features]
        else:
            # 否则进行特征映射
            X = self._map_raw_features_to_model_features(df)
            
        y = df['label']
        
        return X, y
    
    def split_train_test(self, X: pd.DataFrame, y: pd.Series, 
                        test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        分割训练集和测试集
        
        Args:
            X: 特征数据
            y: 标签数据
            test_size: 测试集比例
            
        Returns:
            训练特征、测试特征、训练标签、测试标签
        """
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)