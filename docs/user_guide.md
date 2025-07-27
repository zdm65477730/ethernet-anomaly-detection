# 以太网异常检测系统用户指南

本文档详细介绍了以太网异常检测系统的使用方法，包括完整的操作流程和命令示例。

## 目录

1. [系统概述](#系统概述)
2. [环境准备](#环境准备)
3. [系统使用](#系统使用)
   - [初始化](#初始化)
   - [实时流量检测](#实时流量检测)
   - [离线pcap文件分析](#离线pcap文件分析)
   - [系统状态监控](#系统状态监控)
4. [模型训练与优化](#模型训练与优化)
   - [数据生成](#数据生成)
   - [模型训练](#模型训练)
   - [模型评估](#模型评估)
   - [模型优化](#模型优化)
   - [持续训练](#持续训练)
5. [检测报告](#检测报告)
6. [反馈处理](#反馈处理)

## 系统概述

以太网异常检测系统是一个基于机器学习的网络流量分析工具，能够实时检测网络中的异常行为。系统支持两种工作模式：

1. **实时流量检测** - 监听网络接口并实时分析流量
2. **离线文件分析** - 分析预先捕获的pcap文件

系统采用模块化设计，包含数据捕获、特征提取、异常检测、模型训练等多个组件。

## 环境准备

### 系统要求

- Python 3.8+
- Linux系统（推荐Ubuntu 20.04+）
- 网络接口访问权限

### 安装依赖

```
# 克隆项目
git clone https://github.com/zdm65477730/ethernet-anomaly-detection
cd ethernet-anomaly-detection

# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 系统使用

### 初始化

首次使用系统前需要进行初始化：

```bash
# 初始化系统目录和配置
anomaly-detector init
```

### 实时流量检测

系统支持实时监听网络接口并检测异常流量：

```bash
# 启动系统（监听所有接口）
anomaly-detector start

# 启动系统并指定网络接口
anomaly-detector start --interface eth0

# 启动系统并指定BPF过滤规则
anomaly-detector start --interface eth0 --filter "tcp port 80"
```

### 离线pcap文件分析

系统可以分析预先捕获的pcap文件：

```bash
# 分析pcap文件
anomaly-detector start --pcap-file /path/to/file.pcap
```

### 系统状态监控

查看系统运行状态：

```bash
# 查看系统状态
anomaly-detector status
```

## 模型训练与优化

### 数据生成

生成模拟训练数据用于模型训练：

```bash
# 生成测试数据
anomaly-detector generate-test-data

# 生成指定数量的测试数据
anomaly-detector generate-test-data --count 5000
```

### 模型训练

系统支持多种模型训练方式：

```bash
# 单次训练
anomaly-detector train once

# 指定模型类型和数据进行训练
anomaly-detector train once --model xgboost --data data/processed/model_features_data.csv

# 自动优化训练后的模型
anomaly-detector train once --auto-optimize
```

### 模型评估

评估训练好的模型性能：

```bash
# 评估模型
anomaly-detector train evaluate

# 指定模型类型进行评估
anomaly-detector train evaluate --model xgboost
```

### 模型优化

基于评估结果优化模型和特征工程：

```bash
# 基于最新评估报告进行优化
anomaly-detector train optimize --report reports/xgboost_evaluation_20250727_122142.json

# 基于反馈数据进行优化
anomaly-detector train optimize --feedback-based
```

### 持续训练

启动持续训练模式，系统会自动监测新数据并更新模型：

```bash
# 启动持续训练
anomaly-detector train continuous

# 后台运行持续训练
anomaly-detector train continuous --background
```

### AutoML训练

启动自动化机器学习训练，实现完整的训练-评估-优化-再训练闭环：

```bash
# 启动AutoML训练
anomaly-detector train automl

# 指定数据路径和模型类型
anomaly-detector train automl --data data/processed/ --model xgboost
```

## 检测报告

系统可以生成检测报告和模型评估报告：

```bash
# 生成最近24小时的检测报告
anomaly-detector report generate --last-hours 24

# 生成指定时间范围的检测报告
anomaly-detector report generate --start-time "2025-07-26 00:00:00" --end-time "2025-07-26 23:59:59"

# 生成评估报告
anomaly-detector report generate --type evaluation --last-hours 24

# 生成HTML格式的可视化报告
anomaly-detector report generate --last-hours 24 --format html --visualize
```

### 基于离线数据的检测和报告生成

系统支持对离线pcap文件进行分析并生成图形化检测报告：

```bash
# 1. 使用系统分析离线pcap文件
anomaly-detector start --pcap-file /path/to/network_traffic.pcap

# 2. 等待分析完成（系统会自动停止）

# 3. 生成图形化检测报告
anomaly-detector report generate --last-hours 24 --format html --visualize

# 或者生成特定时间范围的报告
anomaly-detector report generate --start-time "2025-07-27 09:00:00" --end-time "2025-07-27 17:00:00" --format html --visualize
```

### 基于实时网络数据的检测和报告生成

系统也支持实时网络流量检测并生成图形化报告：

```bash
# 1. 启动实时流量检测
anomaly-detector start --interface eth0

# 2. 让系统运行一段时间以收集数据（例如几小时或一天）

# 3. 停止系统
anomaly-detector stop

# 4. 生成图形化检测报告
anomaly-detector report generate --last-hours 24 --format html --visualize
```

### 图形化报告示例

系统生成的HTML格式报告包含以下可视化内容：

1. **检测概览仪表板** - 显示关键指标如总数据包数、会话数、检测到的异常数等
2. **异常趋势图** - 显示随时间变化的异常检测情况
3. **协议分布饼图** - 显示检测到的异常按协议类型的分布
4. **IP地址热力图** - 显示异常活动最频繁的源/目标IP地址
5. **混淆矩阵** - 对于评估报告，显示模型的分类性能
6. **特征重要性柱状图** - 显示各特征对模型预测的贡献度
7. **性能指标雷达图** - 展示模型的各项性能指标

这些图形化报告帮助安全分析师更直观地理解网络异常情况和模型性能。

## 反馈处理

系统支持收集用户对检测结果的反馈，用于优化模型和特征工程。反馈数据保存在 `data/feedback/` 目录中。

### 提交反馈

提交对检测结果的反馈：

```bash
# 标记检测结果为真实异常
anomaly-detector feedback submit --detection-id ALERT_12345 --is-anomaly true --anomaly-type "DDoS"

# 标记检测结果为误报
anomaly-detector feedback submit --detection-id ALERT_12346 --is-anomaly false
```

### 查看反馈

查看已提交的反馈数据：

```bash
# 查看最近的反馈
anomaly-detector feedback list

# 查看最近50条反馈
anomaly-detector feedback list --limit 50
```

### 清理反馈

清理旧的反馈数据：

```bash
# 清理30天前的反馈数据
anomaly-detector feedback cleanup --days 30
```

### 完整的闭环优化流程示例

以下是一个完整的闭环优化流程示例，展示如何通过反馈持续改进模型性能：

**第一次闭环优化：**
```bash
# 1. 生成测试数据
anomaly-detector generate-test-data --count 5000

# 2. 训练模型
anomaly-detector train once --model xgboost --data data/processed/model_features_data.csv

# 3. 评估模型性能
anomaly-detector train evaluate --model xgboost

# 4. 查看生成的评估报告（假定报告名为reports/xgboost_evaluation_20250727_122142.json）
#    根据报告中的检测结果ID提交反馈
anomaly-detector feedback submit --detection-id DETECTION_001 --is-anomaly true --anomaly-type "PortScan"
anomaly-detector feedback submit --detection-id DETECTION_002 --is-anomaly false
anomaly-detector feedback submit --detection-id DETECTION_003 --is-anomaly true --anomaly-type "DDoS"

# 5. 基于反馈优化模型
anomaly-detector train optimize --feedback-based

# 6. 再次训练优化后的模型
anomaly-detector train once --model xgboost --data data/processed/model_features_data.csv
```

**第二次闭环优化：**
```bash
# 1. 启动持续训练模式以自动监测新数据并更新模型
anomaly-detector train continuous --background

# 2. 运行系统一段时间以收集新的网络流量数据
anomaly-detector start --interface eth0

# 3. 等待系统运行一段时间后，检查新生成的检测结果并提交反馈
anomaly-detector feedback submit --detection-id DETECTION_004 --is-anomaly true --anomaly-type "Malware"
anomaly-detector feedback submit --detection-id DETECTION_005 --is-anomaly false

# 4. 停止系统
anomaly-detector stop

# 5. 基于新收集的反馈再次优化模型
anomaly-detector train optimize --feedback-based

# 6. 运行AutoML训练进行全自动优化
anomaly-detector train automl --model xgboost
```

系统采用闭环优化机制，会根据收集到的反馈数据自动优化特征工程和模型参数，持续提升检测准确性。通过多次闭环优化，系统能够不断适应新的网络环境和威胁模式，提高检测准确率并降低误报率。