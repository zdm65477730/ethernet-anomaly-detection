# 异常检测系统架构设计

## 1. 系统概述

本异常检测系统是一个基于机器学习的实时网络流量分析平台，能够自动识别网络中的异常行为和潜在威胁。系统采用模块化设计，支持从数据包捕获、特征提取、模型训练到异常告警的全流程自动化处理，适用于企业内网、数据中心等多种网络环境。

系统核心优势：
- 实时性：毫秒级流量分析与异常检测
- 准确性：结合传统机器学习与深度学习模型
- 自适应性：通过持续学习适应新的网络威胁
- 可扩展性：模块化设计便于功能扩展和定制化开发

## 2. 整体架构

系统采用分层架构设计，共分为6个核心层次，各层次通过标准化接口交互，形成完整的异常检测闭环：

```mermaid
graph TD
    A[数据采集层] --> B[数据处理层]
    B --> C[特征提取层]
    C --> D[模型层]
    D --> E[检测层]
    E --> F[应用层]
    F --> G[基础设施层]
    G --> B
    E --> D
```

- **数据采集层**：负责网络数据包的捕获与解析
- **数据处理层**：处理和存储原始数据与特征数据
- **特征提取层**：从原始流量中提取有价值的特征
- **模型层**：提供模型训练、推理和管理功能
- **检测层**：执行异常检测并触发告警
- **应用层**：提供用户交互接口和告警展示
- **基础设施层**：提供配置管理、日志、监控等支撑功能

## 3. 核心模块详解

### 3.1 数据采集层

**功能**：实时捕获网络流量并解析为结构化数据

**核心组件**：
- `packet_capture.py`：基于libpcap库捕获网络数据包，支持BPF过滤规则
- `session_tracker.py`：维护网络会话状态，聚合数据包信息
- `traffic_analyzer.py`：协调数据包捕获和会话跟踪流程

**关键流程**：
1. 监听指定网络接口（如eth0）
2. 应用BPF过滤规则（可选）
3. 解析数据包为IP/TCP/UDP/ICMP等协议结构
4. 按五元组（源IP、源端口、目的IP、目的端口、协议）聚合为会话
5. 计算会话级统计信息（包数、字节数、持续时间等）

### 3.2 数据处理层

**功能**：存储、清洗和管理流量数据与特征数据

**核心组件**：
- `data_storage.py`：提供数据持久化存储功能
- `data_processor.py`：数据清洗、转换和标准化
- `data_generator.py`：生成训练所需的标注数据集

**数据存储结构**：
- 原始数据包：按时间分区存储，保留完整协议信息
- 会话数据：包含会话元数据和统计信息
- 特征数据：提取的特征向量，用于模型训练和推理
- 标注数据：包含正常/异常标签的数据，用于模型训练

### 3.3 特征提取层

**功能**：从原始流量和会话数据中提取有区分度的特征

**核心组件**：
- `stat_extractor.py`：提取统计特征
- `temporal_extractor.py`：提取时序特征
- `protocol_specs.py`：定义各协议的特征提取规则

**特征类型**：
- 统计特征：包大小分布、协议类型占比、TCP标志位组合等
- 时序特征：包到达间隔、流量速率变化、突发检测等
- 协议专属特征：TCP重传率、UDP payload熵值、ICMP类型分布等
- 聚合特征：多时间窗口（10s/60s/300s）的特征聚合

### 3.4 模型层

**功能**：提供模型训练、推理和管理能力

**核心组件**：
- `base_model.py`：模型基类，定义统一接口
- `traditional_models.py`：传统机器学习模型（XGBoost、随机森林等）
- `deep_models.py`：深度学习模型（LSTM、MLP等）
- `model_factory.py`：模型创建、保存和加载的工厂类
- `model_selector.py`：基于协议类型选择最优模型

**模型管理**：
- 版本控制：每个模型版本保留训练参数和性能指标
- 协议适配：为不同协议类型选择最优模型（如TCP用LSTM，UDP用XGBoost）
- 持续更新：通过增量训练不断优化模型性能

### 3.5 检测层

**功能**：基于模型或规则识别异常流量并触发告警

**核心组件**：
- `anomaly_detector.py`：异常检测核心逻辑
- `alert_manager.py`：告警生成和分发
- `feedback_processor.py`：处理人工反馈，优化检测策略

**检测策略**：
- 模型检测：基于机器学习模型预测异常概率
- 规则检测：基于预定义规则（包大小阈值、异常协议等）
- 混合检测：结合模型和规则，提高检测准确性

### 3.6 应用层与基础设施层

**应用层**：
- `cli/`：命令行接口，支持系统操作和配置
- `webui/`：可选的Web界面，提供可视化展示（未实现）

**基础设施层**：
- `config_manager.py`：配置文件管理
- `logger.py`：日志记录和管理
- `monitor.py`：系统资源监控
- `system_manager.py`：组件生命周期管理

## 4. 核心业务流程

### 4.1 实时检测流程

```mermaid
sequenceDiagram
    participant Capture as 数据包捕获
    participant Session as 会话跟踪
    participant Feature as 特征提取
    participant Detector as 异常检测
    participant Alert as 告警管理
    
    Capture->>Session: 解析后的数据包
    Session->>Feature: 会话数据
    Feature->>Detector: 提取的特征
    Detector->>Alert: 异常结果
    Alert->>Alert: 日志记录 & 邮件通知
```

1. 数据包捕获模块实时捕获并解析网络流量
2. 会话跟踪模块聚合数据包，形成网络会话
3. 特征提取模块从会话中提取统计和时序特征
4. 异常检测模块使用模型预测异常概率
5. 当异常概率超过阈值时，告警管理模块触发告警

### 4.2 模型训练流程

```mermaid
sequenceDiagram
    participant Data as 数据存储
    participant Trainer as 模型训练器
    participant Evaluator as 模型评估器
    participant Factory as 模型工厂
    participant Selector as 模型选择器
    
    Data->>Trainer: 标注数据
    Trainer->>Evaluator: 训练好的模型
    Evaluator->>Selector: 性能指标
    Evaluator->>Factory: 模型 & 性能
    Factory->>Factory: 保存模型版本
    Selector->>Selector: 更新性能记录
```

1. 数据存储模块提供标注数据（正常/异常）
2. 模型训练器进行交叉验证训练
3. 模型评估器计算性能指标（精确率、召回率、F1等）
4. 模型工厂保存模型及版本信息
5. 模型选择器更新性能记录，用于后续模型选择

### 4.3 系统管理流程

```mermaid
sequenceDiagram
    participant CLI as 命令行接口
    participant Manager as 系统管理器
    participant Components as 系统组件
    participant Monitor as 系统监控
    
    CLI->>Manager: 启动系统
    Manager->>Components: 按依赖顺序启动组件
    Components->>Manager: 组件状态报告
    Manager->>Monitor: 启动系统监控
    Monitor->>Manager: 系统状态报告
    
    CLI->>Manager: 停止系统
    Manager->>Manager: 设置停止标志
    Manager->>Components: 按依赖逆序停止组件
    Manager->>Monitor: 停止系统监控
```

1. 系统管理器负责所有组件的生命周期管理
2. 启动时按依赖关系顺序启动各组件
3. 运行时监控各组件状态，必要时自动重启
4. 停止时按依赖关系逆序停止各组件
5. 系统监控持续报告系统资源使用情况

## 5. 系统部署架构

系统支持单机部署和分布式部署两种模式：

- **单机部署**：所有模块运行在同一主机，适用于中小规模网络
- **分布式部署**：各模块可部署在不同主机，通过消息队列通信，适用于大规模网络

最小系统需求：
- CPU：4核及以上
- 内存：8GB及以上
- 存储：至少100GB可用空间
- 操作系统：Linux（推荐Ubuntu 20.04+）

## 6. 组件依赖关系

系统各组件之间存在明确的依赖关系，确保系统稳定运行：

```mermaid
graph TD
    A[PacketCapture] --> B[SessionTracker]
    B --> C[TrafficAnalyzer]
    C --> D[AnomalyDetector]
    D --> E[AlertManager]
    E --> F[FeedbackProcessor]
    C --> G[StatFeatureExtractor]
    C --> H[TemporalFeatureExtractor]
    D --> I[ModelFactory]
    D --> J[ModelSelector]
    K[SystemMonitor] --> L[BaseComponent]
    A --> L
    B --> L
    C --> L
    D --> L
    E --> L
    F --> L
    G --> L
    H --> L
    I --> L
    J --> L
```

组件启动顺序：
1. `SystemMonitor` - 系统监控
2. `PacketCapture` - 数据包捕获
3. `SessionTracker` - 会话跟踪
4. `StatFeatureExtractor` - 统计特征提取
5. `TemporalFeatureExtractor` - 时序特征提取
6. `TrafficAnalyzer` - 流量分析
7. `ModelFactory` - 模型工厂
8. `ModelSelector` - 模型选择器
9. `AnomalyDetector` - 异常检测器
10. `AlertManager` - 告警管理器
11. `FeedbackProcessor` - 反馈处理器

组件停止顺序（与启动顺序相反）：
1. `FeedbackProcessor` - 反馈处理器
2. `AlertManager` - 告警管理器
3. `AnomalyDetector` - 异常检测器
4. `ModelSelector` - 模型选择器
5. `ModelFactory` - 模型工厂
6. `TrafficAnalyzer` - 流量分析
7. `TemporalFeatureExtractor` - 时序特征提取
8. `StatFeatureExtractor` - 统计特征提取
9. `SessionTracker` - 会话跟踪
10. `PacketCapture` - 数据包捕获
11. `SystemMonitor` - 系统监控

## 7. 系统停止机制

系统采用优雅停止机制，确保在停止过程中：

1. **设置停止标志**：SystemManager设置_stopping标志，防止组件重启
2. **按序停止组件**：按照依赖关系逆序停止各组件
3. **等待线程结束**：确保所有后台线程正常终止
4. **释放资源**：各组件清理占用的系统资源
5. **记录日志**：详细记录停止过程中的关键步骤

这种机制确保系统在任何情况下都能安全、完整地停止，避免数据丢失或资源泄露。

## 8. 命令行接口

系统提供统一的命令行接口，所有操作均通过`anomaly-detector`命令执行。

### 8.1 系统管理命令

```bash
# 初始化系统
anomaly-detector init

# 启动系统
anomaly-detector start

# 停止系统
anomaly-detector stop

# 查看系统状态
anomaly-detector status

# 查看当前异常类型分布配置
anomaly-detector config get-anomaly-distribution
```

### 8.2 模型训练命令

```bash
# 单次训练
anomaly-detector train once

# 持续训练
anomaly-detector train continuous

# 评估模型
anomaly-detector train evaluate

# 优化模型
anomaly-detector train optimize

# AutoML训练
anomaly-detector train automl
```

### 8.3 模型管理命令

```bash
# 查看模型列表
anomaly-detector models list

# 切换模型版本
anomaly-detector models use --type xgboost --version 20230615_1230
```

### 8.4 告警管理命令

```bash
# 查看告警列表
anomaly-detector alerts list

# 处理告警反馈
anomaly-detector alerts feedback --alert-id ALERT_12345 --correct
```

### 8.5 反馈处理命令

```bash
# 提交检测结果反馈
anomaly-detector feedback submit --detection-id ALERT_12345 --is-anomaly true

# 查看反馈列表
anomaly-detector feedback list

# 清理旧反馈数据
anomaly-detector feedback cleanup
```

### 8.6 异常分布管理命令

```bash
# 设置异常类型分布
anomaly-detector config set-anomaly-distribution --distribution '{"normal": 0.7, "syn_flood": 0.1, "port_scan": 0.1, "udp_amplification": 0.05, "icmp_flood": 0.05}'

# 启用复合异常生成
anomaly-detector config enable-composite-anomalies

# 禁用复合异常生成
anomaly-detector config disable-composite-anomalies

# 设置复合异常比例
anomaly-detector config set-composite-ratio 0.2
```

### 8.6 报告生成命令

```bash
# 生成检测报告
anomaly-detector report generate --last-hours 24

# 生成评估报告
anomaly-detector report generate --type evaluation --last-hours 24
```

### 8.7 测试数据生成命令

```bash
# 生成测试数据
anomaly-detector generate-test-data
```