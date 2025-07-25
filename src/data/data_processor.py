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
        
        # 特征列信息
        self.numeric_features = [
            "packet_size", "payload_size", "session_duration",
            "inter_arrival_time", "window_size", "retransmissions",
            "packet_size_std", "payload_entropy", "flow_rate",
            "request_response_ratio", "packet_rate", "bytes_per_second"
        ]
        
        self.categorical_features = [
            "protocol", "tcp_flags", "type", "code"
        ]
        
        # 初始化预处理管道
        self._init_preprocessing_pipeline()
    
    def _init_preprocessing_pipeline(self) -> None:
        """初始化数据预处理管道"""
        # 数值特征处理：填充缺失值 + 标准化
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # 分类特征处理：填充缺失值 + 独热编码
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
        
        # 组合所有特征处理器
        self._preprocessing_pipeline = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        self.logger.debug("数据预处理管道已初始化")
    
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
        
        # 2. 处理异常值
        # 处理数值特征中的异常值（如负的包大小）
        for col in self.numeric_features:
            if col in cleaned_df.columns:
                # 对于不能为负的特征，将负值替换为0
                if col in ["packet_size", "payload_size", "retransmissions", "flow_rate"]:
                    cleaned_df[col] = cleaned_df[col].clip(lower=0)
        
        # 3. 处理数据类型
        if "timestamp" in cleaned_df.columns:
            cleaned_df["timestamp"] = pd.to_datetime(cleaned_df["timestamp"], unit='s')
        
        if "label" in cleaned_df.columns:
            cleaned_df["label"] = cleaned_df["label"].astype(int)
        
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
                
                # 返回平衡后的数据集
                balanced_X = X.loc[downsampled_indices]
                balanced_y = y.loc[downsampled_indices]
                
                self.logger.debug(f"平衡后类别分布: {dict(balanced_y.value_counts())}")
                return balanced_X, balanced_y
            else:
                self.logger.warning("少数类样本数为0，无法平衡数据集")
        
        return X, y
    
    def preprocess_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        预处理特征，包括填充缺失值、标准化、编码等
        
        Args:
            df: 特征数据DataFrame
            fit: 是否拟合预处理管道（训练时为True，预测时为False）
        
        Returns:
            预处理后的特征数组
        """
        if df.empty:
            self.logger.warning("尝试预处理空数据，返回空数组")
            return np.array([])
        
        # 选择需要的特征列
        features = df.copy()
        
        # 确保只包含已知的特征列
        all_features = self.numeric_features + self.categorical_features
        available_features = [f for f in all_features if f in features.columns]
        missing_features = [f for f in all_features if f not in features.columns]
        
        if missing_features:
            self.logger.warning(f"缺少以下特征: {missing_features}")
        
        features = features[available_features]
        
        # 拟合或转换
        if fit:
            # 拟合并转换
            processed_features = self._preprocessing_pipeline.fit_transform(features)
            self.logger.debug(f"已拟合预处理管道，处理后特征形状: {processed_features.shape}")
        else:
            # 仅转换
            try:
                processed_features = self._preprocessing_pipeline.transform(features)
                self.logger.debug(f"已转换特征，处理后特征形状: {processed_features.shape}")
            except Exception as e:
                self.logger.error(f"特征转换失败: {str(e)}，尝试重新拟合管道")
                # 失败时尝试重新拟合
                processed_features = self._preprocessing_pipeline.fit_transform(features)
        
        return processed_features
    
    def split_train_test(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray],
                        test_size: Optional[float] = None, random_state: int = 42) -> Tuple[
                            Union[pd.DataFrame, np.ndarray], 
                            Union[pd.DataFrame, np.ndarray], 
                            Union[pd.Series, np.ndarray], 
                            Union[pd.Series, np.ndarray]
                        ]:
        """
        分割训练集和测试集
        
        Args:
            X: 特征数据
            y: 标签数据
            test_size: 测试集比例
            random_state: 随机种子
        
        Returns:
            分割后的训练集和测试集 (X_train, X_test, y_train, y_test)
        """
        if test_size is None:
            test_size = self.config.get("training.test_size", 0.2)
        
        self.logger.debug(f"分割数据集，测试集比例: {test_size}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.logger.info(f"数据集分割完成 - 训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")
        return X_train, X_test, y_train, y_test
    
    def load_raw_data(self, data_dir: str) -> pd.DataFrame:
        """
        加载原始数据
        
        Args:
            data_dir: 数据目录
        
        Returns:
            原始数据DataFrame
        """
        if not os.path.exists(data_dir):
            self.logger.error(f"数据目录不存在: {data_dir}")
            return pd.DataFrame()
        
        # 读取目录中的所有文件
        all_files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith((".parquet", ".csv")):
                    all_files.append(os.path.join(root, file))
        
        if not all_files:
            self.logger.warning(f"在 {data_dir} 中未找到数据文件")
            return pd.DataFrame()
        
        # 读取所有文件并合并
        dfs = []
        for file in all_files:
            try:
                if file.endswith(".parquet"):
                    df = pd.read_parquet(file)
                else:
                    df = pd.read_csv(file)
                dfs.append(df)
                self.logger.debug(f"已加载 {file}，样本数: {len(df)}")
            except Exception as e:
                self.logger.error(f"加载 {file} 失败: {str(e)}")
        
        if not dfs:
            self.logger.warning("所有数据文件均无法加载")
            return pd.DataFrame()
        
        combined_df = pd.concat(dfs, ignore_index=True)
        self.logger.info(f"已加载原始数据，总样本数: {len(combined_df)}")
        return combined_df
    
    def load_processed_data(self, data_dir: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        加载处理后的数据
        
        Args:
            data_dir: 数据目录
        
        Returns:
            特征数据和标签数据 (X, y)
        """
        # 首先尝试从数据存储加载
        if not os.path.isdir(data_dir):
            # 如果不是目录，尝试作为文件加载
            try:
                if data_dir.endswith(".parquet"):
                    df = pd.read_parquet(data_dir)
                else:
                    df = pd.read_csv(data_dir)
                self.logger.info(f"已加载处理后的数据文件: {data_dir}，样本数: {len(df)}")
            except Exception as e:
                self.logger.error(f"加载处理后的数据文件 {data_dir} 失败: {str(e)}")
                return pd.DataFrame(), pd.Series(dtype=int)
        else:
            # 从目录加载
            df = self.load_raw_data(data_dir)
        
        if df.empty:
            return pd.DataFrame(), pd.Series(dtype=int)
        
        # 确保存在标签列
        if "label" not in df.columns:
            self.logger.error("处理后的数据中缺少 'label' 列")
            return pd.DataFrame(), pd.Series(dtype=int)
        
        # 分离特征和标签
        y = df["label"]
        X = df.drop("label", axis=1, errors="ignore")
        
        # 清理数据
        X = self.clean_data(X)
        
        self.logger.info(f"已加载处理后的数据 - 特征数: {X.shape[1]}, 样本数: {X.shape[0]}")
        return X, y
    
    def load_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        加载测试数据
        
        Returns:
            测试特征和标签 (X_test, y_test)
        """
        df = self.storage.load_test_data()
        if df.empty:
            return pd.DataFrame(), pd.Series(dtype=int)
        
        # 确保存在标签列
        if "label" not in df.columns:
            self.logger.error("测试数据中缺少 'label' 列")
            return pd.DataFrame(), pd.Series(dtype=int)
        
        # 分离特征和标签
        y_test = df["label"]
        X_test = df.drop("label", axis=1, errors="ignore")
        
        # 清理数据
        X_test = self.clean_data(X_test)
        
        self.logger.info(f"已加载测试数据 - 特征数: {X_test.shape[1]}, 样本数: {X_test.shape[0]}")
        return X_test, y_test
    
    def load_new_data(self, since: Optional[float] = None) -> pd.DataFrame:
        """
        加载新增的数据（相对于指定时间）
        
        Args:
            since: 起始时间戳，None则加载最近24小时的数据
        
        Returns:
            新增数据DataFrame
        """
        if since is None:
            # 默认加载最近24小时的数据
            since = (datetime.now() - timedelta(days=1)).timestamp()
        
        end_time = datetime.now().timestamp()
        self.logger.info(f"加载 {datetime.fromtimestamp(since)} 至 {datetime.fromtimestamp(end_time)} 的新增数据")
        
        # 从存储加载数据
        new_data = self.storage.load_processed_data_in_range(since, end_time)
        
        if new_data.empty:
            self.logger.info("未发现新增数据")
            return new_data
        
        # 确保数据有标签列
        if "label" not in new_data.columns:
            self.logger.warning("新增数据中缺少 'label' 列，无法用于训练")
            return pd.DataFrame()
        
        self.logger.info(f"已加载新增数据，样本数: {len(new_data)}")
        return new_data
    
    def load_data_by_protocol(self, protocol: Union[str, int]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        加载指定协议的数据
        
        Args:
            protocol: 协议名称或编号
        
        Returns:
            特征和标签 (X, y)
        """
        # 加载所有处理后的数据
        X, y = self.load_processed_data(self.storage.processed_dir)
        if X.empty:
            return X, y
        
        # 筛选指定协议的数据
        if "protocol" not in X.columns:
            self.logger.error("数据中缺少 'protocol' 列，无法按协议筛选")
            return X, y
        
        # 转换协议编号为名称（如果需要）
        from src.features.protocol_specs import get_protocol_spec
        
        if isinstance(protocol, int):
            spec = get_protocol_spec(protocol)
            protocol_name = spec["name"]
            mask = (X["protocol"] == protocol) | (X["protocol"] == protocol_name)
        else:
            mask = X["protocol"] == protocol
        
        X_proto = X[mask]
        y_proto = y[mask]
        
        self.logger.info(f"已加载 {protocol} 协议数据 - 样本数: {len(X_proto)}")
        return X_proto, y_proto
    
    def save_processed_data(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray],
                           timestamp: Optional[float] = None) -> str:
        """
        保存处理后的数据
        
        Args:
            X: 特征数据
            y: 标签数据
            timestamp: 时间戳
        
        Returns:
            保存路径
        """
        # 确保X和y的形状匹配
        if len(X) != len(y):
            self.logger.error(f"特征和标签的样本数不匹配: {len(X)} vs {len(y)}")
            return ""
        
        # 转换为DataFrame
        if isinstance(X, np.ndarray):
            # 如果是numpy数组，转换为DataFrame（假设特征名称已知）
            feature_names = self.numeric_features + self.categorical_features
            # 截取特征名称以匹配数组形状
            if X.shape[1] < len(feature_names):
                feature_names = feature_names[:X.shape[1]]
            elif X.shape[1] > len(feature_names):
                # 如果特征数多于已知特征名称，添加额外的特征名
                feature_names += [f"feature_{i}" for i in range(len(feature_names), X.shape[1])]
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X.copy()
        
        # 添加标签列
        processed_df = X_df.copy()
        processed_df["label"] = y
        
        # 保存数据
        return self.storage.save_processed_data(processed_df, timestamp)
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        # 获取预处理管道中转换后的特征名称
        if self._preprocessing_pipeline is None:
            self._init_preprocessing_pipeline()
            
        feature_names = []
        
        # 获取数值特征名称
        num_features = [f for f in self.numeric_features if f in self._preprocessing_pipeline.get_feature_names_out()]
        feature_names.extend(num_features)
        
        # 获取独热编码后的分类特征名称
        cat_features = self._preprocessing_pipeline.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(self.categorical_features)
        feature_names.extend(cat_features)
        
        return feature_names
    