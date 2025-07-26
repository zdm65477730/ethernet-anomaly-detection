# 模型训练指南

## 1. 概述

异常检测系统支持多种机器学习模型，包括传统机器学习模型（XGBoost、随机森林）和深度学习模型（LSTM、MLP）。本指南详细介绍如何使用系统提供的训练命令来训练、评估和管理模型。

## 2. 训练命令概览

系统提供五种主要的模型训练相关命令:

1. `anomaly-detector train once` - 执行单次模型训练
2. `anomaly-detector train continuous` - 启动持续训练模式
3. `anomaly-detector train evaluate` - 评估现有模型性能
4. `anomaly-detector train optimize` - 基于评估结果优化模型和特征工程
5. `anomaly-detector train automl` - 启动自动化机器学习训练

使用以下命令查看所有训练相关命令的帮助信息：
```bash
anomaly-detector train --help
```

## 3. 单次模型训练

### 3.1 基本用法

执行单次模型训练的最基本命令：
```bash
anomaly-detector train once
```

此命令将使用默认配置和数据目录中的数据训练一个XGBoost模型。

### 3.2 指定模型类型

可以使用`--model`或`-m`参数指定要训练的模型类型：
```bash
# 训练随机森林模型
anomaly-detector train once --model random_forest

# 训练LSTM模型
anomaly-detector train once --model lstm
```

支持的模型类型包括：
- `xgboost` - XGBoost模型（默认）
- `random_forest` - 随机森林模型
- `logistic_regression` - 逻辑回归模型
- `lstm` - LSTM神经网络模型
- `mlp` - 多层感知机模型

### 3.3 指定数据路径

使用`--data`或`-d`参数指定训练数据路径：
```bash
anomaly-detector train once --data /path/to/training/data
```

### 3.4 设置测试集比例

使用`--test-size`或`-t`参数设置测试集比例（0-1之间）：
```bash
anomaly-detector train once --test-size 0.3
```

### 3.5 使用交叉验证

使用`--cv`或`-k`参数指定交叉验证折数：
```bash
anomaly-detector train once --cv 5
```

### 3.6 指定输出目录

使用`--output`或`-o`参数指定模型输出目录：
```bash
anomaly-detector train once --output /path/to/model/output
```

### 3.7 自动优化

使用`--auto-optimize`或`-a`参数在训练后自动进行优化：
```bash
anomaly-detector train once --auto-optimize
```

### 3.8 完整示例

```bash
anomaly-detector train once \
  --model xgboost \
  --data ./data/processed \
  --test-size 0.2 \
  --cv 5 \
  --output ./models \
  --auto-optimize \
  --log-level INFO
```

## 4. 持续训练模式

### 4.1 启动持续训练

持续训练模式会定期检查新数据并增量更新模型：
```bash
anomaly-detector train continuous
```

### 4.2 设置检查间隔

使用`--interval`或`-i`参数设置检查新数据的时间间隔（秒）：
```bash
anomaly-detector train continuous --interval 7200  # 每2小时检查一次
```

### 4.3 设置最小样本数

使用`--min-samples`或`-m`参数设置触发训练的最小样本数：
```bash
anomaly-detector train continuous --min-samples 5000
```

### 4.4 后台运行

使用`--background`或`-b`参数在后台运行持续训练：
```bash
anomaly-detector train continuous --background
```

### 4.5 完整示例

```bash
anomaly-detector train continuous \
  --interval 3600 \
  --min-samples 1000 \
  --background \
  --log-level INFO
```

## 5. 模型评估

### 5.1 评估最新模型

评估指定类型的最新模型：
```bash
anomaly-detector train evaluate --type xgboost
```

### 5.2 评估指定模型

使用`--model`或`-m`参数指定要评估的模型文件路径：
```bash
anomaly-detector train evaluate --model /path/to/model/file.pkl
```

### 5.3 指定测试数据

使用`--data`或`-d`参数指定测试数据路径：
```bash
anomaly-detector train evaluate --data /path/to/test/data
```

### 5.4 指定输出路径

使用`--output`或`-o`参数指定评估报告输出路径：
```bash
anomaly-detector train evaluate --output /path/to/report.json
```

### 5.5 自动优化

使用`--auto-optimize`或`-a`参数在评估后自动进行优化：
```bash
anomaly-detector train evaluate --auto-optimize
```

### 5.6 完整示例

```bash
anomaly-detector train evaluate \
  --type xgboost \
  --data ./data/test \
  --output ./reports/evaluation_report.json \
  --auto-optimize
```

## 6. 模型优化

### 6.1 基于评估报告优化

使用`anomaly-detector train optimize`命令基于评估报告进行优化：
```bash
anomaly-detector train optimize --model xgboost --report /path/to/evaluation_report.json
```

### 6.2 优化建议

优化命令会根据模型性能提供以下类型的建议：
- 特征工程优化（移除低重要性特征，加强高重要性特征工程）
- 模型参数调整（增加模型复杂度，调整训练参数）
- 重新训练建议（当性能过低时建议全量重训）

## 7. AutoML自动化训练

### 7.1 启动AutoML训练

AutoML训练会自动执行多轮训练、评估、优化过程。它会自动检测训练数据中的协议类型，并为每种协议选择最适合的模型：
```bash
anomaly-detector train automl
```

### 7.2 指定初始模型类型

使用`--model`或`-m`参数指定初始模型类型：
```bash
anomaly-detector train automl --model lstm
```

### 7.3 指定协议类型

使用`--protocol`或`-p`参数指定协议类型（系统会自动为该协议选择最佳模型）：
```bash
anomaly-detector train automl --protocol 6  # TCP协议
```

### 7.4 指定数据路径

使用`--data`或`-d`参数指定训练数据路径：
```bash
anomaly-detector train automl --data /path/to/training/data
```

### 7.5 后台运行

使用`--background`或`-b`参数在后台运行AutoML训练：
```bash
anomaly-detector train automl --background
```

### 7.6 完整示例

```bash
anomaly-detector train automl \
  --data ./data/processed \
  --background \
  --log-level INFO
```

## 8. 模型管理

### 8.1 查看模型列表

查看所有可用模型：
```bash
anomaly-detector train models list
```

查看指定类型的模型：
```bash
anomaly-detector train models list --type xgboost
```

### 8.2 切换模型版本

切换到指定版本的模型：
```bash
anomaly-detector train models use --type xgboost --version 20230615_1230
```

## 9. 训练数据准备

### 9.1 数据格式

训练数据应为预处理后的特征数据，通常存储在以下格式中：
- CSV文件，每行代表一个样本
- 必须包含'label'列作为目标变量（0表示正常，1表示异常）
- 可选包含'protocol'或'protocol_num'列标识协议类型
- 其余列为特征变量

### 9.2 数据目录结构

推荐的数据目录结构：
```
data/
├── raw/              # 原始流量数据
├── processed/        # 预处理后的特征数据
└── test/             # 测试数据
```

### 9.3 生成模拟数据

如果需要生成模拟训练数据进行测试：
```bash
anomaly-detector train once
# 当提示数据路径不存在时，选择生成模拟数据
```

## 10. 反馈优化机制

### 10.1 自动优化流程

系统支持基于模型评估结果的自动优化机制，包括：

1. **性能评估**：检查模型各项指标是否达到阈值
2. **特征分析**：分析特征重要性，识别低价值和高价值特征
3. **参数建议**：根据性能提供模型参数调整建议
4. **优化执行**：生成优化建议报告

### 10.2 特征工程优化

系统会根据特征重要性提供以下优化建议：
- 移除低重要性特征（重要性<1%）
- 加强高重要性特征的工程（重要性>10%）
- 调整特征提取参数

### 10.3 模型参数优化

根据模型性能提供参数调整建议：
- **XGBoost/RandomForest**：增大n_estimators或max_depth
- **LSTM**：增加隐藏层维度或训练轮数
- **MLP**：增加层数或神经元数量

### 10.4 持续优化

在持续训练模式下，系统会自动执行以下优化流程：
1. 模型训练完成后自动评估性能
2. 基于评估结果生成优化建议
3. 记录优化历史供后续分析
4. 提供可视化建议供人工决策

## 11. 最佳实践

### 11.1 模型选择建议

- **XGBoost**: 适用于大多数场景，训练速度快，性能稳定
- **随机森林**: 鲁棒性强，适合处理噪声数据
- **LSTM**: 适用于具有强时序特性的流量数据
- **MLP**: 适用于复杂的非线性关系建模

### 11.2 训练参数调优

- 测试集比例通常设置为0.2-0.3
- 交叉验证折数建议为5-10折
- 根据数据量调整模型复杂度参数

### 11.3 持续训练配置

- 检查间隔根据数据产生速度设置（1-24小时）
- 最小样本数根据业务场景设置（1000-10000）
- 建议在业务低峰期启动持续训练

### 11.4 模型评估指标

重点关注以下指标：
- **准确率(Accuracy)**: 整体预测正确的比例
- **精确率(Precision)**: 预测为异常中实际异常的比例
- **召回率(Recall)**: 实际异常中被正确预测的比例
- **F1分数**: 精确率和召回率的调和平均数

## 12. 故障排除

### 12.1 训练失败

常见原因及解决方案：
- 数据路径错误：检查数据路径是否存在
- 内存不足：减少批量大小或使用更简单的模型
- 权限问题：确保有足够的文件读写权限

### 12.2 模型性能不佳

- 检查训练数据质量和数量
- 调整模型超参数
- 尝试不同的特征工程方法
- 增加训练数据的多样性

### 12.3 持续训练问题

- 检查数据目录是否有新数据写入
- 确认系统资源是否充足
- 查看日志文件了解详细错误信息

### 12.4 优化建议问题

- 检查评估报告是否包含特征重要性信息
- 确认模型类型是否支持特征重要性分析
- 查看优化历史了解优化效果