import os
import time
import shutil
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any
from src.utils.logger import get_logger
from src.config.config_manager import ConfigManager

class DataStorage:
    """数据存储管理器，负责数据的分区存储和查询"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        初始化数据存储管理器
        
        Args:
            config: 配置管理器
        """
        self.config = config or ConfigManager()
        self.logger = get_logger("data_storage")
        
        # 数据目录
        self.raw_dir = self.config.get("data.raw_dir", "data/raw")
        self.processed_dir = self.config.get("data.processed_dir", "data/processed")
        self.test_dir = self.config.get("data.test_dir", "data/test")
        
        # 存储格式
        self.storage_format = self.config.get("data.storage_format", "parquet")
        if self.storage_format not in ["parquet", "csv"]:
            self.logger.warning(f"不支持的存储格式 {self.storage_format}，使用默认格式 parquet")
            self.storage_format = "parquet"
        
        # 数据保留天数
        self.retention_days = self.config.get("data.retention_days", 30)
        
        # 初始化目录
        self._init_directories()
        
        # 启动数据清理线程
        self._start_cleanup_thread()
    
    def _init_directories(self) -> None:
        """初始化数据目录"""
        for dir_path in [self.raw_dir, self.processed_dir, self.test_dir]:
            os.makedirs(dir_path, exist_ok=True)
            self.logger.debug(f"初始化数据目录: {dir_path}")
    
    def _get_partition_path(self, base_dir: str, timestamp: Optional[float] = None) -> str:
        """
        获取分区路径（按年月分区）
        
        Args:
            base_dir: 基础目录
            timestamp: 时间戳，None则使用当前时间
        
        Returns:
            分区路径
        """
        if timestamp is None:
            timestamp = time.time()
        dt = datetime.fromtimestamp(timestamp)
        partition = f"{dt.year}{dt.month:02d}"  # 格式: YYYYMM
        return os.path.join(base_dir, partition)
    
    def _get_filename(self, timestamp: Optional[float] = None) -> str:
        """
        生成文件名（按日期时间）
        
        Args:
            timestamp: 时间戳，None则使用当前时间
        
        Returns:
            文件名
        """
        if timestamp is None:
            timestamp = time.time()
        dt = datetime.fromtimestamp(timestamp)
        filename = f"{dt.year}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}{dt.minute:02d}.{self.storage_format}"
        return filename
    
    def save_raw_data(self, data: pd.DataFrame, timestamp: Optional[float] = None) -> str:
        """
        保存原始数据
        
        Args:
            data: 原始数据DataFrame
            timestamp: 时间戳
        
        Returns:
            保存路径
        """
        return self._save_data(data, self.raw_dir, timestamp)
    
    def save_processed_data(self, data: pd.DataFrame, timestamp: Optional[float] = None) -> str:
        """
        保存处理后的数据
        
        Args:
            data: 处理后的数据DataFrame
            timestamp: 时间戳
        
        Returns:
            保存路径
        """
        return self._save_data(data, self.processed_dir, timestamp)
    
    def save_test_data(self, data: pd.DataFrame) -> str:
        """
        保存测试数据
        
        Args:
            data: 测试数据DataFrame
        
        Returns:
            保存路径
        """
        # 测试数据不分区，直接保存
        os.makedirs(self.test_dir, exist_ok=True)
        filename = f"test_data.{self.storage_format}"
        path = os.path.join(self.test_dir, filename)
        
        self._write_data(data, path)
        self.logger.info(f"已保存测试数据至 {path}，样本数: {len(data)}")
        return path
    
    def _save_data(self, data: pd.DataFrame, base_dir: str, timestamp: Optional[float] = None) -> str:
        """
        保存数据到指定目录
        
        Args:
            data: 数据DataFrame
            base_dir: 基础目录
            timestamp: 时间戳
        
        Returns:
            保存路径
        """
        if data.empty:
            self.logger.warning("尝试保存空数据，操作已跳过")
            return ""
        
        partition_path = self._get_partition_path(base_dir, timestamp)
        os.makedirs(partition_path, exist_ok=True)
        
        filename = self._get_filename(timestamp)
        path = os.path.join(partition_path, filename)
        
        self._write_data(data, path)
        self.logger.debug(f"已保存数据至 {path}，样本数: {len(data)}")
        return path
    
    def _write_data(self, data: pd.DataFrame, path: str) -> None:
        """
        写入数据到文件
        
        Args:
            data: 数据DataFrame
            path: 文件路径
        """
        try:
            if self.storage_format == "parquet":
                data.to_parquet(path, index=False)
            else:  # csv
                data.to_csv(path, index=False)
        except Exception as e:
            self.logger.error(f"写入数据到 {path} 失败: {str(e)}", exc_info=True)
            raise
    
    def _read_data(self, path: str) -> pd.DataFrame:
        """
        从文件读取数据
        
        Args:
            path: 文件路径
        
        Returns:
            数据DataFrame
        """
        try:
            if path.endswith(".parquet"):
                return pd.read_parquet(path)
            elif path.endswith(".csv"):
                return pd.read_csv(path)
            else:
                self.logger.error(f"不支持的文件格式: {path}")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"从 {path} 读取数据失败: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    def count_new_data_since(self, timestamp: float, protocol: Optional[int] = None) -> int:
        """
        统计指定时间戳之后的新数据数量
        
        Args:
            timestamp: 时间戳阈值
            protocol: 协议类型（可选）
            
        Returns:
            新数据的数量
        """
        try:
            # 加载处理后的数据（在实际应用中可能需要优化）
            data = self.load_processed_data_in_range(
                start_time=timestamp,
                end_time=time.time()
            )
            
            # 如果指定了协议，进行过滤
            if protocol is not None:
                # 假设数据中有一个protocol列
                if 'protocol' in data.columns:
                    data = data[data['protocol'] == protocol]
            
            return len(data)
        except Exception as e:
            self.logger.warning(f"统计新数据数量时出错: {e}")
            return 0
    
    def get_files_in_range(self, base_dir: str, start_time: float, end_time: float) -> List[str]:
        """
        获取指定时间范围内的文件
        
        Args:
            base_dir: 基础目录
            start_time: 开始时间戳
            end_time: 结束时间戳
        
        Returns:
            文件路径列表
        """
        files = []
        
        # 计算需要检查的分区范围
        start_dt = datetime.fromtimestamp(start_time)
        end_dt = datetime.fromtimestamp(end_time)
        
        # 生成所有需要检查的年月分区
        current_dt = start_dt
        partitions = set()
        while current_dt <= end_dt:
            partition = f"{current_dt.year}{current_dt.month:02d}"
            partitions.add(partition)
            # 移动到下个月
            if current_dt.month == 12:
                current_dt = current_dt.replace(year=current_dt.year + 1, month=1)
            else:
                current_dt = current_dt.replace(month=current_dt.month + 1)
        
        # 检查每个分区
        for partition in partitions:
            partition_path = os.path.join(base_dir, partition)
            if not os.path.exists(partition_path):
                continue
            
            # 检查分区中的文件
            for filename in os.listdir(partition_path):
                if not (filename.endswith(f".{self.storage_format}") or 
                        filename.endswith(".parquet") or 
                        filename.endswith(".csv")):
                    continue
                
                # 解析文件名中的日期
                try:
                    # 文件名格式: YYYYMMDD_HHMM.format
                    date_str = filename.split(".")[0]
                    file_dt = datetime.strptime(date_str, "%Y%m%d_%H%M")
                    file_timestamp = file_dt.timestamp()
                    
                    # 检查是否在时间范围内
                    if start_time <= file_timestamp <= end_time:
                        files.append(os.path.join(partition_path, filename))
                except Exception as e:
                    self.logger.warning(f"解析文件名 {filename} 失败: {str(e)}")
                    continue
        
        # 按时间排序
        files.sort()
        return files
    
    def load_raw_data_in_range(self, start_time: float, end_time: float) -> pd.DataFrame:
        """
        加载指定时间范围内的原始数据
        
        Args:
            start_time: 开始时间戳
            end_time: 结束时间戳
        
        Returns:
            合并后的DataFrame
        """
        return self._load_data_in_range(self.raw_dir, start_time, end_time)
    
    def load_processed_data_in_range(self, start_time: float, end_time: float) -> pd.DataFrame:
        """
        加载指定时间范围内的处理后数据
        
        Args:
            start_time: 开始时间戳
            end_time: 结束时间戳
        
        Returns:
            合并后的DataFrame
        """
        return self._load_data_in_range(self.processed_dir, start_time, end_time)
    
    def _load_data_in_range(self, base_dir: str, start_time: float, end_time: float) -> pd.DataFrame:
        """
        加载指定目录和时间范围内的数据
        
        Args:
            base_dir: 基础目录
            start_time: 开始时间戳
            end_time: 结束时间戳
        
        Returns:
            合并后的DataFrame
        """
        if start_time >= end_time:
            self.logger.warning("开始时间必须小于结束时间")
            return pd.DataFrame()
        
        files = self.get_files_in_range(base_dir, start_time, end_time)
        if not files:
            self.logger.info(f"在 {base_dir} 中未找到 {datetime.fromtimestamp(start_time)} 至 {datetime.fromtimestamp(end_time)} 之间的数据")
            return pd.DataFrame()
        
        self.logger.info(f"加载 {len(files)} 个文件，总时间范围: {datetime.fromtimestamp(start_time)} 至 {datetime.fromtimestamp(end_time)}")
        
        # 读取并合并所有文件
        dfs = []
        for file in files:
            df = self._read_data(file)
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            self.logger.warning("所有文件均为空或无法读取")
            return pd.DataFrame()
        
        # 合并数据
        combined_df = pd.concat(dfs, ignore_index=True)
        self.logger.info(f"已加载数据，总样本数: {len(combined_df)}")
        return combined_df
    
    def load_test_data(self) -> pd.DataFrame:
        """
        加载测试数据
        
        Returns:
            测试数据DataFrame
        """
        test_file = os.path.join(self.test_dir, f"test_data.{self.storage_format}")
        if not os.path.exists(test_file):
            # 检查是否有其他格式的测试数据
            for ext in ["parquet", "csv"]:
                alt_file = os.path.join(self.test_dir, f"test_data.{ext}")
                if os.path.exists(alt_file):
                    test_file = alt_file
                    break
            else:
                # 如果test目录中没有测试数据，检查processed目录
                processed_test_file = os.path.join(self.processed_dir, "test_data.csv")
                if os.path.exists(processed_test_file):
                    test_file = processed_test_file
                else:
                    self.logger.warning(f"未找到测试数据文件: {test_file}")
                    return pd.DataFrame()
        
        df = self._read_data(test_file)
        self.logger.info(f"已加载测试数据，样本数: {len(df)}")
        return df
    
    def _cleanup_old_data(self) -> None:
        """清理过期数据"""
        if self.retention_days <= 0:
            self.logger.debug("数据保留天数设置为0，不清理数据")
            return
        
        cutoff_time = time.time() - (self.retention_days * 86400)  # 86400秒 = 1天
        cutoff_dt = datetime.fromtimestamp(cutoff_time)
        self.logger.info(f"开始清理 {cutoff_dt} 之前的数据")
        
        # 需要清理的目录
        dirs_to_clean = [self.raw_dir, self.processed_dir]
        
        for base_dir in dirs_to_clean:
            if not os.path.exists(base_dir):
                continue
            
            # 检查每个分区
            for partition in os.listdir(base_dir):
                partition_path = os.path.join(base_dir, partition)
                if not os.path.isdir(partition_path):
                    continue
                
                # 解析分区年月
                try:
                    # 分区格式: YYYYMM
                    partition_dt = datetime.strptime(partition, "%Y%m")
                    # 检查分区是否过期（整个分区的月份早于截止时间的月份）
                    if (partition_dt.year < cutoff_dt.year or 
                        (partition_dt.year == cutoff_dt.year and partition_dt.month < cutoff_dt.month)):
                        self.logger.info(f"清理过期分区: {partition_path}")
                        shutil.rmtree(partition_path)
                except Exception as e:
                    self.logger.warning(f"解析分区 {partition} 失败: {str(e)}，跳过清理")
                    continue
        
        self.logger.info("数据清理完成")
    
    def _start_cleanup_thread(self) -> None:
        """启动数据清理线程（每天执行一次）"""
        import threading
        
        def cleanup_loop():
            while True:
                # 等待24小时
                time.sleep(86400)
                try:
                    self._cleanup_old_data()
                except Exception as e:
                    self.logger.error(f"数据清理线程出错: {str(e)}", exc_info=True)
        
        self.cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        self.logger.debug("数据清理线程已启动")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        def get_dir_size(path: str) -> int:
            """计算目录大小（字节）"""
            total = 0
            for dirpath, _, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp):
                        total += os.path.getsize(fp)
            return total
        
        # 计算各目录大小
        raw_size = get_dir_size(self.raw_dir)
        processed_size = get_dir_size(self.processed_dir)
        test_size = get_dir_size(self.test_dir)
        
        # 计算文件数量
        def count_files(path: str) -> int:
            count = 0
            for _, _, filenames in os.walk(path):
                count += len(filenames)
            return count
        
        raw_files = count_files(self.raw_dir)
        processed_files = count_files(self.processed_dir)
        test_files = count_files(self.test_dir)
        
        return {
            "storage_format": self.storage_format,
            "retention_days": self.retention_days,
            "directories": {
                "raw": {
                    "path": self.raw_dir,
                    "size_bytes": raw_size,
                    "size_human": self._format_size(raw_size),
                    "file_count": raw_files
                },
                "processed": {
                    "path": self.processed_dir,
                    "size_bytes": processed_size,
                    "size_human": self._format_size(processed_size),
                    "file_count": processed_files
                },
                "test": {
                    "path": self.test_dir,
                    "size_bytes": test_size,
                    "size_human": self._format_size(test_size),
                    "file_count": test_files
                }
            },
            "total": {
                "size_bytes": raw_size + processed_size + test_size,
                "size_human": self._format_size(raw_size + processed_size + test_size),
                "file_count": raw_files + processed_files + test_files
            }
        }
    
    def _format_size(self, size_bytes: int) -> str:
        """将字节数格式化为人类可读的字符串"""
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        size = size_bytes
        unit_index = 0
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        return f"{size:.2f} {units[unit_index]}"
    
    def __str__(self) -> str:
        """返回存储管理器的字符串表示"""
        stats = self.get_storage_stats()
        return (f"DataStorage(format={self.storage_format}, "
                f"total_size={stats['total']['size_human']}, "
                f"total_files={stats['total']['file_count']})")
    