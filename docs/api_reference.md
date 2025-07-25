# 异常检测系统API参考文档

## 1. 概述

本文档详细描述异常检测系统的核心API接口，涵盖类定义、方法参数、返回值及使用示例。系统采用面向对象设计模式，各模块通过标准化接口交互，确保扩展性与易用性。所有API均基于Python 3.8+实现，兼容主流操作系统（Linux/macOS）。

## 2. 数据采集模块（`src/capture/`）

### 2.1 `PacketCapture`类
**功能**：实时捕获并解析网络数据包，支持协议解析与过滤。

#### 初始化
```python
from src.capture.packet_capture import PacketCapture

capture = PacketCapture(
    interface: str = "eth0",        # 网络接口名称
    filter: str = "",               # BPF过滤规则（如"tcp port 80"）
    buffer_size: int = 65535,       # 抓包缓冲区大小（字节）
    timeout: int = 100              # 抓包超时时间（毫秒）
)
```

#### 核心方法

**`start()`**
```python
def start(self) -> None:
    """启动抓包线程，开始监听网络接口"""
```

**`stop()`**
```python
def stop(self) -> None:
    """停止抓包线程，释放网络接口资源"""
```

**`get_packet()`**
```python
def get_packet(
    self,
    block: bool = True,             # 是否阻塞等待数据包
    timeout: Optional[float] = None # 超时时间（秒）
) -> Optional[Dict]:
    """
    获取解析后的数据包
    
    返回: 数据包字典（含协议字段）或None（超时）
    """
```

**数据包结构示例**：
```python
{
    "timestamp": 1620000000.123456,  # 时间戳（秒）
    "src_ip": "192.168.1.100",      # 源IP地址
    "dst_ip": "8.8.8.8",             # 目的IP地址
    "src_port": 12345,              # 源端口
    "dst_port": 80,                 # 目的端口
    "protocol": 6,                  # 协议编号（6=TCP,17=UDP,1=ICMP）
    "length": 1500,                 # 数据包长度（字节）
    "tcp_flags": {                   # TCP标志位（仅TCP协议存在）
        "SYN": True,
        "ACK": False,
        "FIN": False
    }
}
```


### 2.2 `SessionTracker`类
**功能**：按网络会话聚合数据包，维护会话状态与统计信息。

#### 初始化
```python
from src.capture.session_tracker import SessionTracker

tracker = SessionTracker(
    timeout: int = 300  # 会话超时时间（秒），无活动则自动清理
)
```

#### 核心方法

**`track_packet()`**
```python
def track_packet(self, packet: Dict) -> Tuple[str, Dict]:
    """
    跟踪数据包并更新会话状态
    
    参数:
        packet: 解析后的数据包字典（来自PacketCapture）
    
    返回:
        会话ID（五元组哈希）和会话信息字典
    """
```

**`get_session()`**
```python
def get_session(self, session_id: str) -> Optional[Dict]:
    """
    获取指定会话的详细信息
    
    返回: 包含包数、字节数、持续时间等的会话字典
    """
```

**`cleanup_expired()`**
```python
def cleanup_expired(self) -> int:
    """
    清理超时会话
    
    返回: 被清理的会话数量
    """
```


## 3. 特征提取模块（`src/features/`）

### 3.1 `StatFeatureExtractor`类
**功能**：从会话数据中提取统计特征（如包大小分布、协议占比等）。

#### 核心方法
```python
from src.features.stat_extractor import StatFeatureExtractor

extractor = StatFeatureExtractor()

def extract_features_from_session(self, session: Dict) -> Dict[str, float]:
    """
    提取会话的统计特征
    
    参数:
        session: 会话信息字典（来自SessionTracker）
    
    返回:
        特征字典，示例：
        {
            "packet_count": 42,            # 包总数
            "total_bytes": 15600,          # 总字节数
            "avg_packet_size": 371.43,     # 平均包大小
            "tcp_syn_count": 1,            # TCP SYN包数量
            "protocol_ratio": 0.85         # 主协议占比
        }
    """
```


### 3.2 `TemporalFeatureExtractor`类
**功能**：提取时序特征（如流量速率、包到达间隔等），支持多时间窗口。

#### 初始化
```python
from src.features.temporal_extractor import TemporalFeatureExtractor

extractor = TemporalFeatureExtractor(
    window_sizes: List[int] = [10, 60, 300]  # 时间窗口（秒）
)
```

#### 核心方法
```python
def extract_features_from_session(self, session: Dict) -> Dict[str, float]:
    """
    提取会话的时序特征
    
    返回:
        特征字典，示例：
        {
            "packet_rate_10s": 5.2,        # 10秒窗口内的包速率
            "byte_rate_60s": 1200.5,       # 60秒窗口内的字节速率
            "inter_arrival_std": 0.87      # 包到达间隔标准差
        }
    """
```


## 4. 模型模块（`src/models/`）

### 4.1 `ModelFactory`类
**功能**：模型工厂类，负责模型的创建、保存、加载与版本管理。

#### 核心方法
```python
from src.models.model_factory import ModelFactory

factory = ModelFactory()

def create_model(self, model_type: str, **kwargs) -> BaseModel:
    """
    创建指定类型的模型实例
    
    参数:
        model_type: 模型类型，支持：
            - "xgboost"：XGBoost模型
            - "random_forest"：随机森林
            - "logistic_regression"：逻辑回归
            - "lstm"：长短期记忆网络
            - "mlp"：多层感知器
       ** kwargs: 模型参数（见model_config.yaml）
    
    返回:
        模型实例（继承自BaseModel）
    """

def save_model(self, model: BaseModel) -> str:
    """
    保存模型至文件系统
    
    返回: 模型保存路径（含时间戳与版本信息）
    """

def load_latest_model(self, model_type: str) -> BaseModel:
    """加载指定类型的最新模型版本"""
```


### 4.2 `BaseModel`抽象类
**功能**：所有模型的基类，定义统一接口（训练/预测/评估）。

#### 核心方法
```python
from src.models.base_model import BaseModel

# 所有模型均实现以下接口
def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
    """
    训练模型
    
    参数:
        X: 特征矩阵（n_samples × n_features）
        y: 标签数组（0=正常，1=异常）
       ** kwargs: 训练参数（如epochs、batch_size）
    """

def predict(self, X: np.ndarray) -> np.ndarray:
    """
    预测标签
    
    返回: 预测结果数组（0或1）
    """

def predict_proba(self, X: np.ndarray) -> np.ndarray:
    """
    预测异常概率
    
    返回: 概率数组（形状为(n_samples, 2)，[:,1]为异常概率）
    """

def save(self, file_path: str) -> None:
    """保存模型至指定路径"""

@classmethod
def load(cls, file_path: str) -> BaseModel:
    """从文件加载模型"""
```


### 4.3 `ModelSelector`类
**功能**：基于协议类型与历史性能自动选择最优模型。

#### 核心方法
```python
from src.models.model_selector import ModelSelector

selector = ModelSelector()

def select_best_model(self, protocol: Union[str, int]) -> str:
    """
    为指定协议选择最优模型
    
    参数:
        protocol: 协议名称（如"tcp"）或编号（如6）
    
    返回:
        最优模型类型（如"lstm"）
    """

def update_performance(self, protocol: Union[str, int], model_type: str, metrics: Dict[str, float]) -> None:
    """
    更新模型性能记录（用于后续选择）
    
    参数:
        metrics: 性能指标，含"f1"、"precision"、"recall"等
    """
```


## 5. 异常检测模块（`src/detection/`）

### 5.1 `AnomalyDetector`类
**功能**：结合模型预测与规则引擎检测异常流量。

#### 初始化
```python
from src.detection.anomaly_detector import AnomalyDetector

detector = AnomalyDetector(
    threshold: float = 0.8,  # 异常判定阈值（概率≥此值视为异常）
    mode: str = "hybrid"    # 检测模式："model"/"rule"/"hybrid"
)
```

#### 核心方法
```python
def detect(self, features: Dict[str, float]) -> Tuple[bool, float]:
    """
    检测异常流量
    
    参数:
        features: 特征字典（来自特征提取模块）
    
    返回:
        (是否异常, 异常概率)
    """
```


### 5.2 `AlertManager`类
**功能**：处理异常检测结果，触发日志记录与告警通知。

#### 核心方法
```python
from src.detection.alert_manager import AlertManager

alert_manager = AlertManager()

def trigger_alert(self, features: Dict[str, float], score: float, session_id: str) -> None:
    """
    触发异常告警
    
    参数:
        features: 特征字典
        score: 异常概率
        session_id: 会话ID
    """
```


## 6. 训练模块（`src/training/`）

### 6.1 `ContinuousTrainer`类
**功能**：实现持续训练逻辑，自动检测新数据并更新模型。

#### 核心方法
```python
from src.training.continuous_trainer import ContinuousTrainer

trainer = ContinuousTrainer()

def start(self, check_interval: int = 3600, min_samples: int = 1000) -> None:
    """
    启动持续训练进程
    
    参数:
        check_interval: 检查新数据的时间间隔（秒）
        min_samples: 触发训练的最小新样本数
    """

def trigger_manual_training(self, model_type: str, protocol: Optional[int] = None) -> Tuple[bool, str]:
    """
    手动触发模型训练
    
    参数:
        model_type: 模型类型
        protocol: 协议编号（None表示通用模型）
    
    返回:
        (训练是否成功, 结果信息)
    """
```


### 6.2 `ModelTrainer`类
**功能**：基础训练逻辑，支持交叉验证与模型评估。

#### 核心方法
```python
from src.training.model_trainer import ModelTrainer

trainer = ModelTrainer(model_factory=ModelFactory())

def train_new_model(self, model_type: str, X: np.ndarray, y: np.ndarray) -> Tuple[BaseModel, Dict[str, float]]:
    """
    训练新模型
    
    参数:
        model_type: 模型类型
        X: 特征矩阵
        y: 标签数组
    
    返回:
        (训练好的模型, 评估指标)
    """
```


## 7. 配置管理（`src/config/`）

### 7.1 `ConfigManager`类
**功能**：管理系统配置参数，支持动态读写配置文件。

#### 核心方法
```python
from src.config.config_manager import ConfigManager

config = ConfigManager()

def get(self, key: str, default: Any = None) -> Any:
    """
    获取配置参数
    
    参数:
        key: 参数键（支持点语法，如"model.threshold"）
        default: 默认值
    """

def set(self, key: str, value: Any) -> None:
    """更新配置参数"""
```


## 8. 工具函数（`src/utils/`）

### 8.1 数据处理工具
```python
from src.utils.helpers import (
    load_json,               # 加载JSON文件
    save_json,               # 保存JSON文件
    split_train_test,        # 分割训练集/测试集
    balance_dataset,         # 平衡数据集（处理类别不平衡）
    normalize_features       # 特征标准化
)
```

### 8.2 评估工具
```python
from src.utils.metrics import (
    calculate_precision,     # 计算精确率
    calculate_recall,        # 计算召回率
    calculate_f1,           # 计算F1分数
    calculate_auc,           # 计算AUC值
    calculate_confusion_matrix  # 计算混淆矩阵
)
```

### 8.3 可视化工具
```python
from src.utils.evaluation_visualizer import EvaluationVisualizer

visualizer = EvaluationVisualizer(output_dir="plots/")

def plot_roc_curve(self, y_true: np.ndarray, y_score: np.ndarray) -> str:
    """绘制ROC曲线并返回保存路径"""

def plot_confusion_matrix(self, cm: np.ndarray) -> str:
    """绘制混淆矩阵热图"""
```


## 9. 错误处理与日志

所有API方法均会抛出以下异常（需捕获处理）：
- `ValueError`：参数无效或数据格式错误
- `FileNotFoundError`：模型/配置文件不存在
- `RuntimeError`：运行时错误（如训练失败）

日志可通过`src.utils.logger`模块获取，按模块分类输出至`logs/`目录。