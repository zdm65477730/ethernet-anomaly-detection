from typing import Any, List, Dict, Optional, Tuple
import os
import json
import pickle
import time
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Dict, List, Union, Optional
import socket
import fcntl
import struct
from collections import Counter

def standardize_data(
    X_train: Union[np.ndarray, pd.DataFrame],
    X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None
) -> Tuple[Union[np.ndarray, pd.DataFrame], Optional[Union[np.ndarray, pd.DataFrame]], StandardScaler]:
    """
    标准化数据 (均值为0，标准差为1)
    
    参数:
        X_train: 训练数据
        X_test: 测试数据 (可选)
        
    返回:
        标准化后的训练数据、测试数据和标准化器
    """
    scaler = StandardScaler()
    
    # 处理DataFrame
    if isinstance(X_train, pd.DataFrame):
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        if X_test is not None and isinstance(X_test, pd.DataFrame):
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        else:
            X_test_scaled = None
    else:
        # 处理numpy数组
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    
    return X_train_scaled, X_test_scaled, scaler

def normalize_data(
    X_train: Union[np.ndarray, pd.DataFrame],
    X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    feature_range: Tuple[float, float] = (0, 1)
) -> Tuple[Union[np.ndarray, pd.DataFrame], Optional[Union[np.ndarray, pd.DataFrame]], MinMaxScaler]:
    """
    归一化数据 (缩放到指定范围)
    
    参数:
        X_train: 训练数据
        X_test: 测试数据 (可选)
        feature_range: 缩放范围
        
    返回:
        归一化后的训练数据、测试数据和归一化器
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    
    if isinstance(X_train, pd.DataFrame):
        X_train_normalized = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        if X_test is not None and isinstance(X_test, pd.DataFrame):
            X_test_normalized = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        else:
            X_test_normalized = None
    else:
        X_train_normalized = scaler.fit_transform(X_train)
        X_test_normalized = scaler.transform(X_test) if X_test is not None else None
    
    return X_train_normalized, X_test_normalized, scaler

def calculate_entropy(data: bytes) -> float:
    """
    计算字节数据的熵值（衡量随机性）
    
    参数:
        data: 字节数据
        
    返回:
        熵值 (0-8之间，8表示完全随机)
    """
    if not data:
        return 0.0
    
    # 统计每个字节的出现频率
    byte_counts = Counter(data)
    data_length = len(data)
    
    # 计算熵值
    entropy = 0.0
    for count in byte_counts.values():
        probability = count / data_length
        entropy -= probability * np.log2(probability)
    
    return entropy

def balance_dataset(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    method: str = "oversample"
) -> Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.Series]]:
    """
    平衡数据集 (处理类别不平衡问题)
    
    参数:
        X: 特征数据
        y: 标签数据
        method: 平衡方法 ("oversample" 或 "undersample")
        
    返回:
        平衡后的特征和标签
    """
    # 统计类别分布
    unique, counts = np.unique(y, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    max_count = max(counts)
    min_count = min(counts)
    
    # 如果已经平衡，直接返回
    if max_count == min_count:
        return X, y
    
    balanced_X = []
    balanced_y = []
    
    for cls in unique:
        # 获取该类别的样本索引
        cls_mask = y == cls
        cls_X = X[cls_mask]
        cls_y = y[cls_mask]
        cls_count = len(cls_X)
        
        if method == "oversample":
            # 过采样：复制少数类样本
            if cls_count < max_count:
                # 计算需要复制的次数和剩余样本数
                repeat = max_count // cls_count
                remainder = max_count % cls_count
                
                # 复制样本
                oversampled_X = np.repeat(cls_X, repeat, axis=0)
                oversampled_y = np.repeat(cls_y, repeat, axis=0)
                
                # 添加剩余样本
                if remainder > 0:
                    remainder_indices = np.random.choice(cls_count, remainder, replace=False)
                    oversampled_X = np.concatenate([oversampled_X, cls_X[remainder_indices]])
                    oversampled_y = np.concatenate([oversampled_y, cls_y[remainder_indices]])
                
                balanced_X.append(oversampled_X)
                balanced_y.append(oversampled_y)
            else:
                # 多数类直接添加
                balanced_X.append(cls_X)
                balanced_y.append(cls_y)
        
        elif method == "undersample":
            # 欠采样：随机减少多数类样本
            if cls_count > min_count:
                # 随机选择与少数类相同数量的样本
                undersample_indices = np.random.choice(cls_count, min_count, replace=False)
                undersampled_X = cls_X[undersample_indices]
                undersampled_y = cls_y[undersample_indices]
                
                balanced_X.append(undersampled_X)
                balanced_y.append(undersampled_y)
            else:
                # 少数类直接添加
                balanced_X.append(cls_X)
                balanced_y.append(cls_y)
        
        else:
            raise ValueError(f"不支持的平衡方法: {method}，请使用 'oversample' 或 'undersample'")
    
    # 合并所有类别的样本
    combined_X = np.concatenate(balanced_X, axis=0)
    combined_y = np.concatenate(balanced_y, axis=0)
    
    # 打乱顺序
    shuffle_indices = np.random.permutation(len(combined_X))
    combined_X = combined_X[shuffle_indices]
    combined_y = combined_y[shuffle_indices]
    
    # 如果是DataFrame，保持结构
    if isinstance(X, pd.DataFrame):
        combined_X = pd.DataFrame(
            combined_X, 
            columns=X.columns
        )
    if isinstance(y, pd.Series):
        combined_y = pd.Series(
            combined_y, 
            name=y.name
        )
    
    return combined_X, combined_y

def train_test_split(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[
    Union[np.ndarray, pd.DataFrame],
    Union[np.ndarray, pd.DataFrame],
    Union[np.ndarray, pd.Series],
    Union[np.ndarray, pd.Series]
]:
    """
    划分训练集和测试集
    
    参数:
        X: 特征数据
        y: 标签数据
        test_size: 测试集比例
        random_state: 随机种子
        stratify: 是否按标签分层抽样
        
    返回:
        划分后的训练特征、测试特征、训练标签、测试标签
    """
    stratify_param = y if stratify else None
    return sk_train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )

def save_json(data: Dict, file_path: str, indent: int = 2) -> bool:
    """
    保存数据到JSON文件
    
    参数:
        data: 要保存的数据
        file_path: 文件路径
        indent: JSON缩进
        
    返回:
        保存是否成功
    """
    try:
        # 创建目录
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception as e:
        from .logger import get_logger
        logger = get_logger("utils.helpers")
        logger.error(f"保存JSON文件 {file_path} 失败: {str(e)}")
        return False

def load_json(file_path: str) -> Optional[Dict]:
    """
    从JSON文件加载数据
    
    参数:
        file_path: 文件路径
        
    返回:
        加载的数据，失败则返回None
    """
    try:
        if not os.path.exists(file_path):
            return None
            
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        from .logger import get_logger
        logger = get_logger("utils.helpers")
        logger.error(f"加载JSON文件 {file_path} 失败: {str(e)}")
        return None

def save_pickle(data: Any, file_path: str) -> bool:
    """
    保存数据到Pickle文件
    
    参数:
        data: 要保存的数据
        file_path: 文件路径
        
    返回:
        保存是否成功
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return True
    except Exception as e:
        from .logger import get_logger
        logger = get_logger("utils.helpers")
        logger.error(f"保存Pickle文件 {file_path} 失败: {str(e)}")
        return False

def load_pickle(file_path: str) -> Optional[Any]:
    """
    从Pickle文件加载数据
    
    参数:
        file_path: 文件路径
        
    返回:
        加载的数据，失败则返回None
    """
    try:
        if not os.path.exists(file_path):
            return None
            
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        from .logger import get_logger
        logger = get_logger("utils.helpers")
        logger.error(f"加载Pickle文件 {file_path} 失败: {str(e)}")
        return None

def timestamp_to_str(timestamp: float, format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    将时间戳转换为字符串
    
    参数:
        timestamp: 时间戳
        format: 时间格式
        
    返回:
        格式化的时间字符串
    """
    return time.strftime(format, time.localtime(timestamp))

def str_to_timestamp(time_str: str, format: str = "%Y-%m-%d %H:%M:%S") -> float:
    """
    将时间字符串转换为时间戳
    
    参数:
        time_str: 时间字符串
        format: 时间格式
        
    返回:
        时间戳
    """
    return time.mktime(time.strptime(time_str, format))

def is_valid_ip(ip: str) -> bool:
    """
    检查IP地址是否有效
    
    参数:
        ip: IP地址字符串
        
    返回:
        是否有效的布尔值
    """
    # 支持IPv4和IPv6
    try:
        socket.inet_pton(socket.AF_INET, ip)
        return True
    except OSError:
        try:
            socket.inet_pton(socket.AF_INET6, ip)
            return True
        except OSError:
            return False

def get_ip_addresses(interface: Optional[str] = None) -> Dict[str, str]:
    """
    获取本机IP地址
    
    参数:
        interface: 网络接口名称，为None则获取所有
        
    返回:
        接口名称到IP地址的映射
    """
    ip_addresses = {}
    
    # 遍历所有网络接口
    for ifname in os.listdir('/sys/class/net/'):
        if interface and ifname != interface:
            continue
            
        try:
            # 创建原始套接字
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # 获取IP地址
            ip_addr = socket.inet_ntoa(fcntl.ioctl(
                sock.fileno(),
                0x8915,  # SIOCGIFADDR
                struct.pack('256s', ifname.encode('utf-8')[:15])
            )[20:24])
            
            ip_addresses[ifname] = ip_addr
        except Exception as e:
            from .logger import get_logger
            logger = get_logger("utils.helpers")
            logger.debug(f"获取接口 {ifname} 的IP地址失败: {str(e)}")
    
    return ip_addresses

def parse_packet_timestamp(timestamp_str: str) -> float:
    """
    解析数据包时间戳字符串为时间戳
    
    参数:
        timestamp_str: 数据包时间戳字符串
        
    返回:
        时间戳
    """
    # 支持多种时间格式
    formats = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S.%f",
        "%m/%d/%Y %H:%M:%S"
    ]
    
    for fmt in formats:
        try:
            return time.mktime(time.strptime(timestamp_str, fmt))
        except ValueError:
            continue
    
    # 如果所有格式都失败，尝试其他解析方式
    try:
        # 尝试Unix时间戳字符串
        return float(timestamp_str)
    except ValueError:
        from .logger import get_logger
        logger = get_logger("utils.helpers")
        logger.warning(f"无法解析时间戳: {timestamp_str}")
        return time.time()  # 返回当前时间作为 fallback
