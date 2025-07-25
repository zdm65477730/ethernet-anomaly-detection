# 异常检测系统用户操作指南

## 1. 系统简介

异常检测系统是一款基于机器学习的网络安全防护工具，能够实时监控网络流量、识别异常行为并自动触发告警。系统融合了传统机器学习与深度学习技术，可适应企业内网、数据中心等多种网络环境，帮助安全人员快速发现潜在威胁。

系统核心特性：
- 实时流量分析（毫秒级响应）
- 自适应学习（动态优化检测模型）
- 多维度检测（结合模型预测与规则引擎）
- 灵活部署（支持单机与分布式架构）

## 2. 环境准备与安装

### 2.1 环境要求

| 类别 | 最低配置 | 推荐配置 |
|------|----------|----------|
| 操作系统 | Linux (Ubuntu 18.04+/CentOS 7+) | Ubuntu 20.04 LTS |
| Python | 3.8.x | 3.9.x |
| CPU | 4核 | 8核 |
| 内存 | 8GB | 16GB |
| 存储 | 100GB可用空间 | 500GB SSD |
| 网络 | 1Gbps网卡 | 10Gbps网卡 |

### 2.2 安装步骤

#### 步骤1：克隆代码仓库# 克隆仓库

```bash
git clone https://github.com/your-org/anomaly-detection.git
cd anomaly-detection
```

# 切换到稳定版本
```bash
git checkout v1.0.0
```

#### 步骤2：创建并激活虚拟环境# 创建虚拟环境
```bash
# 如果没有创建过，需要创建，创建过不需要再次创建。
python3 -m venv venv

# 激活环境（Linux/macOS）
source venv/bin/activate

# 激活环境（Windows PowerShell）
.\venv\Scripts\Activate.ps1
```

#### 步骤3：安装依赖包# 安装基础依赖
```bash
# https://mirrors.aliyun.com/pypi/simple/ or https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
pip install -r requirements_dev.txt -i https://mirrors.aliyun.com/pypi/simple
```

# 如需使用深度学习模型（可选）
```bash
pip install -r requirements_gpu.txt -i https://mirrors.aliyun.com/pypi/simple # 含GPU支持
```

# 或
```bash
pip install -r requirements_cpu.txt  # 仅CPU支持
```

#### 步骤4：系统初始化# 初始化配置文件和目录结构
```bash
python -m src.cli init
```

# （可选）导入初始样本数据（用于模型预热）
```bash
python -m src.cli import_samples --path ./samples/initial_data.csv
```

初始化成功后会生成以下目录结构：anomaly-detection/
├── config/               # 配置文件
├── data/                 # 数据存储
│   ├── raw/              # 原始流量数据
│   ├── features/         # 提取的特征数据
│   └── models/           # 训练好的模型
├── logs/                 # 系统日志
└── src/                  # 源代码

## 3. 系统配置

### 3.1 配置文件说明
系统配置文件集中在`config/`目录，核心文件包括：

| 文件名 | 功能描述 |
|--------|----------|
| `config.yaml` | 主配置文件（网络、告警等全局设置） |
| `model_config.yaml` | 模型参数配置（算法、训练参数等） |
| `detection_rules.yaml` | 规则引擎配置（阈值、异常模式等） |

### 3.2 基础配置（config.yaml）# 网络监听配置
network:
  interface: "eth0"         # 监听的网络接口（使用ip link查看可用接口）
  filter: ""                # BPF过滤规则（如"tcp port 80 or udp port 53"）
  capture_limit: 1000       # 每秒最大抓包数（0表示无限制）

# 模型配置
model:
  default_type: "xgboost"   # 默认模型类型（xgboost/lstm/random_forest）
  threshold: 0.8            # 异常判定阈值（0-1之间）
  update_interval: 86400    # 模型自动更新间隔（秒）

# 告警配置
alert:
  enabled: true             # 是否启用告警
  level: "medium"           # 告警级别（low/medium/high）
  cooldown: 300             # 告警冷却时间（秒，避免重复告警）
  log_to_file: true         # 是否记录告警日志
  email_notify: false       # 是否启用邮件通知

### 3.3 邮件告警配置
如需启用邮件通知，补充以下配置：alert:
  # ... 其他配置 ...
  email_notify: true
  smtp:
    server: "smtp.example.com"
    port: 587
    username: "alerts@example.com"
    password: "your-app-password"
    use_tls: true
    recipients:
      - "security-team@example.com"
      - "admin@example.com"

### 3.4 模型参数配置（model_config.yaml）
根据网络环境调整模型参数：# XGBoost模型配置（适用于TCP/UDP流量）
xgboost:
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1
  subsample: 0.8

# LSTM模型配置（适用于时序性强的流量）
lstm:
  input_dim: 32
  hidden_dim: 64
  num_layers: 2
  dropout: 0.2
  epochs: 50
  batch_size: 128
## 4. 系统操作指南

### 4.1 启动系统# 基础启动（使用默认配置）
```bash
python -m src.cli start
```

# 指定网络接口启动
```bash
python -m src.cli start -i eth1
```

# 启用详细日志模式
```bash
python -m src.cli start -v
```

# 使用自定义过滤规则（仅监控HTTP/HTTPS流量）
```bash
python -m src.cli start -f "tcp port 80 or tcp port 443"
```

系统启动成功后，会在后台运行并输出以下信息：[INFO] 系统启动成功，进程ID: 12345
[INFO] 监听接口: eth0
[INFO] 加载模型: xgboost (v20230615)
[INFO] 状态监控地址: http://localhost:8080/status

### 4.2 查看系统状态# 查看基本状态
```bash
python -m src.cli status
```

# 查看详细统计信息
```bash
python -m src.cli status --detail
```

# 查看实时流量指标
```bash
python -m src.cli metrics
```

状态信息示例：系统状态: 运行中 (已运行 2小时15分)
组件状态:
  - 数据包捕获: 正常 (速率: 124 包/秒)
  - 特征提取: 正常 (延迟: 8ms)
  - 异常检测: 正常 (准确率: 98.2%)
  - 告警系统: 正常 (今日告警: 3 条)
资源使用:
  - CPU: 15.3%
  - 内存: 2.4GB
  - 磁盘: 12.7GB (数据)

### 4.3 停止系统# 正常停止（优雅关闭）
```bash
python -m src.cli stop
```

# 强制停止（用于无响应情况）
```bash
python -m src.cli stop --force
```

### 4.4 模型管理

#### 查看模型列表python -m src.cli models list
#### 手动触发模型训练# 训练通用模型（使用所有协议数据）
```bash
python -m src.cli train once
```

# 针对特定协议训练（如TCP协议，编号6）
```bash
python -m src.cli train once --protocol 6
```

# 训练指定类型模型
```bash
python -m src.cli train once --model lstm
```

# 全量重训（使用历史所有数据）
```bash
python -m src.cli train once --full-retrain
```

#### 切换模型版本# 查看可用版本
```bash
python -m src.cli models list --type xgboost
```

# 切换到指定版本
```bash
python -m src.cli models use --type xgboost --version 20230615_1230
```

### 4.5 日志管理

#### 查看实时日志# 查看系统主日志
```bash
tail -f logs/system.log
```

# 查看检测日志
```bash
tail -f logs/detection.log
```

# 查看告警日志
```bash
tail -f logs/alerts.log
```

#### 日志轮转配置
系统默认启用日志轮转，配置文件位于`config/logrotate.conf`，可设置保留天数和大小限制：max_size = 100M    # 单个日志文件最大大小
max_days = 30      # 日志保留天数
compress = true    # 是否压缩旧日志

## 5. 告警处理

### 5.1 告警级别与含义
| 级别 | 含义 | 处理建议 |
|------|------|----------|
| 低（Low） | 可疑行为但风险较低（如罕见端口通信） | 记录观察，无需立即处理 |
| 中（Medium） | 可能存在异常（如流量突增、协议异常） | 人工核查，确认是否为误报 |
| 高（High） | 确认异常或攻击行为（如端口扫描、数据泄露） | 立即处理，采取阻断措施 |

### 5.2 查看告警历史# 查看最近10条告警
```bash
python -m src.cli alerts list --limit 10
```

# 查看特定级别告警
```bash
python -m src.cli alerts list --level high
```

# 导出告警为CSV
```bash
python -m src.cli alerts export --path ./alerts_202306.csv
```

### 5.3 告警反馈（优化模型）
对误报或漏报进行反馈，帮助系统优化：# 标记某条告警为误报
```bash
python -m src.cli alerts feedback --id 123 --result false_positive
```

# 标记某条正常记录为漏报
```bash
python -m src.cli alerts feedback --session-id "192.168.1.100:12345-8.8.8.8:80" --result false_negative
```

## 6. 日常维护

### 6.1 数据清理# 清理30天前的原始数据包
```bash
python -m src.cli cleanup --data raw --days 30
```

# 清理特征数据（保留最近15天）
```bash
python -m src.cli cleanup --data features --days 15
```

# 清理所有过期数据（按配置自动判断）
```bash
python -m src.cli cleanup --all
```

### 6.2 系统备份# 备份配置文件
```bash
python -m src.cli backup --config
```

# 备份模型文件
```bash
python -m src.cli backup --models
```

# 完整备份（配置+模型+关键数据）
```bash
python -m src.cli backup --full --path /backup/anomaly_detection/$(date +%Y%m%d)
```

### 6.3 系统升级# 停止当前系统
```bash
python -m src.cli stop
```

# 拉取最新代码
```bash
git pull
git checkout v1.1.0  # 切换到新版本
```

# 更新依赖
```bash
pip install -r requirements.txt --upgrade
```

# 升级配置文件（保留自定义设置）
```bash
python -m src.cli upgrade config
```

# 启动新版本
```bash
python -m src.cli start
```

## 7. 常见问题与解决方案

### 7.1 系统启动失败
| 错误现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 权限不足 | 无网络接口访问权限 | 使用sudo运行或添加CAP_NET_RAW权限 |
| 接口不存在 | 配置的网络接口不存在 | 执行`ip link`查看可用接口，修改config.yaml |
| 依赖缺失 | 未安装全部依赖 | 重新执行`pip install -r requirements.txt` |
| 端口占用 | 监控端口被占用 | 检查8080端口占用情况：`lsof -i:8080`并释放 |

### 7.2 检测准确率低
- **解决方案1**：重新训练模型
  ```bash
  python -m src.cli train once --full-retrain
  ```

- **解决方案2**：调整检测阈值
  ```yaml
  # 在config.yaml中修改
  model:
    threshold: 0.75  # 降低阈值提高召回率（可能增加误报）
    # 或提高阈值降低误报（可能增加漏报）
  ```

- **解决方案3**：补充标注数据
  ```bash
  # 导入带标签的样本数据
  python -m src.cli import_samples --path ./new_labeled_data.csv
  ```

### 7.3 系统资源占用过高
- 降低抓包速率限制：
  ```yaml
  network:
    capture_limit: 500  # 限制每秒最大抓包数
  ```

- 简化特征提取：
  ```yaml
  # 在model_config.yaml中
  features:
    reduced_mode: true  # 启用精简特征集
  ```

- 调整模型复杂度：
  ```yaml
  # 降低XGBoost复杂度
  xgboost:
    n_estimators: 50
    max_depth: 4
  ```

### 7.4 告警风暴（大量重复告警）
- 增加告警冷却时间：
  ```yaml
  alert:
    cooldown: 600  # 10分钟内同类型告警只触发一次
  ```

- 提高告警级别阈值：
  ```yaml
  alert:
    level: "high"  # 只处理高级别告警
  ```

- 添加抑制规则：
  ```yaml
  # 在detection_rules.yaml中
  suppress_rules:
    - name: "已知服务器通信"
      condition: "dst_ip in ['192.168.1.1', '192.168.1.2']"
      level: "low"  # 抑制低级别告警
  ```

## 8. 联系方式与支持

- 文档中心：https://docs.example.com/anomaly-detection
- 问题反馈：support@example.com
- 社区论坛：https://forum.example.com/c/anomaly-detection
- 紧急支持：400-123-4567（工作日9:00-18:00）
