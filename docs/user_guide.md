# 异常检测系统用户指南

## 1. 系统概述

异常检测系统是一款基于机器学习的网络安全防护工具，能够实时监控网络流量、识别异常行为并自动触发告警。系统融合了传统机器学习与深度学习技术，可适应企业内网、数据中心等多种网络环境。

### 1.1 功能特点

- 实时流量分析（毫秒级处理网络数据包）
- 自适应学习（动态优化检测模型）
- 多维度检测（结合模型预测与规则引擎）
- 灵活部署（支持单机与分布式架构）

### 1.2 系统架构

系统采用分层设计，主要包含以下模块：
1. **数据采集层**：捕获并解析网络数据包
2. **特征提取层**：从流量中提取有价值的特征
3. **模型层**：多种机器学习模型用于异常检测
4. **检测层**：结合模型预测与规则引擎判断异常
5. **应用层**：提供CLI接口和监控面板

## 2. 环境准备与安装

### 2.1 系统要求

- 操作系统：Linux (Ubuntu 20.04+/CentOS 8+)
- Python：3.8 或更高版本
- 内存：至少4GB可用内存
- 存储：至少10GB可用磁盘空间

### 2.2 安装步骤

#### 方法一：快速安装（推荐）
```bash
# 克隆仓库
git clone https://github.com/zdm65477730/ethernet-anomaly-detection.git
cd ethernet-anomaly-detection

# 使用辅助脚本安装依赖
chmod +x scripts/install_dependencies.sh
sudo ./scripts/install_dependencies.sh

# 以开发模式安装
pip install -e .
```

#### 方法二：手动安装
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装生产依赖
pip install -r requirements.txt

# 安装开发依赖（可选）
pip install -r requirements_dev.txt

# 安装项目
pip install -e .
```

### 2.3 初始化系统

安装完成后，需要初始化系统配置和目录结构：
```bash
anomaly-detector init
```

此命令将创建以下目录结构：
```
.
├── config/          # 配置文件目录
├── data/            # 数据存储目录
│   ├── raw/         # 原始流量数据
│   ├── processed/   # 处理后的特征数据
│   └── test/        # 测试数据
├── logs/            # 日志文件目录
├── models/          # 模型文件目录
├── reports/         # 报告文件目录
└── alerts/          # 告警记录目录
```

## 3. 系统使用指南

### 3.1 启动系统

#### 基本启动
```bash
anomaly-detector start
```

#### 指定网络接口
```bash
# 监听特定网络接口（如eth1）
anomaly-detector start --interface eth1
```

#### 启用详细日志模式
```bash
anomaly-detector start --verbose
```

#### 使用自定义过滤规则（仅监控HTTP/HTTPS流量）
```bash
anomaly-detector start --filter "tcp port 80 or tcp port 443"
```

系统启动成功后，会在后台运行并输出以下信息：
```
[INFO] 系统启动成功，进程ID: 12345
[INFO] 监听接口: eth0
[INFO] 加载模型: xgboost (v20230615)
[INFO] 状态监控地址: http://localhost:8080/status
```

### 3.2 查看系统状态

#### 查看基本状态
```bash
anomaly-detector status
```

#### 查看详细统计信息
```bash
anomaly-detector status --detail
```

#### 查看实时流量指标
```bash
anomaly-detector metrics
```

状态信息示例：
```
系统状态: 运行中 (已运行 2小时15分)
组件状态:
  - 数据包捕获: 正常 (速率: 124 包/秒)
  - 特征提取: 正常 (延迟: 8ms)
  - 异常检测: 正常 (准确率: 98.2%)
  - 告警系统: 正常 (今日告警: 3 条)
资源使用:
  - CPU: 15.3%
  - 内存: 2.4GB
  - 磁盘: 12.7GB (数据)
```

### 3.3 停止系统

#### 正常停止（优雅关闭）
```bash
anomaly-detector stop
```

#### 强制停止（用于无响应情况）
```bash
anomaly-detector stop --force
```

### 3.4 模型管理

#### 查看模型列表
```bash
anomaly-detector models list
```

#### 手动触发模型训练
```bash
# 训练通用模型（使用所有协议数据）
anomaly-detector train once

# 针对特定协议训练（如TCP协议，编号6）
anomaly-detector train once --protocol 6

# 训练指定类型模型
anomaly-detector train once --model lstm

# 全量重训（使用历史所有数据）
anomaly-detector train once --full-retrain
```

#### 持续训练模式
```bash
# 启动持续训练模式
anomaly-detector train continuous

# 查看持续训练帮助
anomaly-detector train continuous --help
```

#### 模型评估
```bash
# 评估现有模型
anomaly-detector train evaluate

# 查看评估帮助
anomaly-detector train evaluate --help
```

#### 切换模型版本
```bash
# 查看可用版本
anomaly-detector models list --type xgboost

# 切换到指定版本
anomaly-detector models use --type xgboost --version 20230615_1230
```

### 3.5 日志管理

#### 查看实时日志
```bash
# 查看系统主日志
tail -f logs/system.log

# 查看检测日志
tail -f logs/detection.log

# 查看告警日志
tail -f logs/alerts.log
```

#### 日志轮转
系统默认每天轮转日志文件，保留最近30天的日志。可以通过修改配置文件调整轮转策略。

### 3.6 告警处理

#### 查看历史告警
```bash
anomaly-detector alerts list
```

#### 查看特定时间段告警
```bash
anomaly-detector alerts list --start-time "2023-06-01 00:00:00" --end-time "2023-06-02 00:00:00"
```

#### 导出告警报告
```bash
anomaly-detector alerts export --format json --output alerts_report.json
```

#### 告警反馈
```bash
# 标记告警为正确检测
anomaly-detector alerts feedback --alert-id ALERT_12345 --correct

# 标记告警为误报
anomaly-detector alerts feedback --alert-id ALERT_12345 --incorrect
```

## 4. 配置管理

### 4.1 配置文件结构

系统配置文件位于`config/`目录，主要配置文件包括：
- `config.yaml`：主配置文件（网络、告警等）
- `model_config.yaml`：模型参数配置
- `detection_rules.yaml`：规则引擎配置

### 4.2 查看当前配置
```bash
anomaly-detector config show
```

### 4.3 修改配置
可以直接编辑配置文件，或使用命令行工具：
```bash
# 设置网络接口
anomaly-detector config set network.interface eth1

# 设置检测阈值
anomaly-detector config set detection.threshold 0.85

# 设置日志级别
anomaly-detector config set general.log_level DEBUG
```

### 4.4 重载配置
修改配置文件后，可以重载配置而无需重启系统：
```bash
anomaly-detector config reload
```

## 5. 日常维护

### 5.1 数据清理

#### 清理旧数据
```bash
# 清理30天前的原始数据
anomaly-detector data cleanup --days 30 --type raw

# 清理所有类型数据
anomaly-detector data cleanup --days 7
```

#### 数据备份
```bash
# 备份模型数据
anomaly-detector data backup --type models --output backup_models_20230615.tar.gz

# 备份所有数据
anomaly-detector data backup --output full_backup_20230615.tar.gz
```

### 5.2 系统监控

#### 查看系统健康状态
```bash
anomaly-detector health
```

#### 性能基准测试
```bash
anomaly-detector benchmark
```

#### 资源使用情况
```bash
anomaly-detector top
```

### 5.3 模型更新

#### 手动更新模型
```bash
# 从外部文件更新模型
anomaly-detector models update --type xgboost --file new_xgboost_model.pkl

# 从模型库更新
anomaly-detector models update --type lstm --version latest
```

#### 模型性能报告
```bash
anomaly-detector models report
```

## 6. 故障排除

### 6.1 系统启动失败

#### 检查依赖项
```bash
# 检查Python版本
python --version

# 检查必需库
pip list | grep -E "(numpy|pandas|scikit-learn|xgboost)"
```

#### 检查权限
```bash
# 确保有网络接口访问权限
sudo setcap cap_net_raw,cap_net_admin=eip /usr/bin/python3
```

#### 检查配置
```bash
# 验证配置文件语法
anomaly-detector config validate
```

### 6.2 检测性能不佳

#### 检查模型状态
```bash
# 查看模型版本和性能
anomaly-detector models list --detail

# 评估当前模型
anomaly-detector train evaluate
```

#### 调整检测参数
```bash
# 降低阈值以提高敏感性
anomaly-detector config set detection.threshold 0.6

# 启用详细日志以诊断问题
anomaly-detector config set general.log_level DEBUG
```

### 6.3 误报过多

#### 分析告警反馈
```bash
# 查看最近的误报
anomaly-detector alerts list --type false-positive --limit 10

# 分析误报模式
anomaly-detector alerts analyze --type false-positive
```

#### 优化检测规则
```bash
# 添加白名单规则
anomaly-detector config set detection.whitelist.1 "src_ip: 192.168.1.1"

# 调整规则阈值
anomaly-detector config set detection.rules.icmp_flood_threshold 20
```

### 6.4 性能问题

#### 降低资源消耗
```bash
# 简化特征提取：
# 在model_config.yaml中添加
features:
  reduced_mode: true  # 启用精简特征集

# 调整模型复杂度：
# 在model_config.yaml中添加
xgboost:
  n_estimators: 50
  max_depth: 4
```

### 6.5 告警风暴（大量重复告警）
```bash
# 增加告警冷却时间：
# 在config.yaml中添加
alert:
  cooldown: 600  # 10分钟内同类型告警只触发一次

# 提高告警级别阈值：
# 在config.yaml中添加
alert:
  level: "high"  # 只处理高级别告警

# 添加抑制规则：
# 在detection_rules.yaml中添加
suppress_rules:
  - name: "已知服务器通信"
    condition: "dst_ip in ['192.168.1.1', '192.168.1.2']"
    level: "low"  # 抑制低级别告警
```

## 7. 联系方式与支持

- 文档中心：https://docs.example.com/anomaly-detection
- 问题反馈：support@example.com
- 社区论坛：https://forum.example.com/c/anomaly-detection
- 紧急支持：400-123-4567（工作日9:00-18:00）