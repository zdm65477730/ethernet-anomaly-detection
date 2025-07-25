# 异常检测系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/zdm65477730/ethernet-anomaly-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/zdm65477730/ethernet-anomaly-detection/actions/workflows/tests.yml)

异常检测系统是一款基于机器学习的网络安全防护工具，能够实时监控网络流量、识别异常行为并自动触发告警。系统融合了传统机器学习与深度学习技术，可适应企业内网、数据中心等多种网络环境。

## 功能特点

- **实时流量分析**：毫秒级处理网络数据包，不影响网络性能
- **多维度检测**：结合统计特征、时序模式和协议分析
- **自适应学习**：动态更新模型以适应网络环境变化
- **灵活部署**：支持单机部署和分布式架构
- **可扩展架构**：易于添加新的检测算法和协议解析器

## 系统架构

![系统架构图](docs/diagrams/architecture.png)

系统采用分层设计，主要包含以下模块：
1. **数据采集层**：捕获并解析网络数据包
2. **特征提取层**：从流量中提取有价值的特征
3. **模型层**：多种机器学习模型用于异常检测
4. **检测层**：结合模型预测与规则引擎判断异常
5. **应用层**：提供CLI接口和监控面板

## 安装指南

### 前置要求

- 操作系统：Linux (Ubuntu 18.04+/CentOS 7+)
- Python：3.8 或更高版本
- 依赖库：详见requirements.txt

### 快速安装
# 克隆仓库
git clone https://github.com/zdm65477730/ethernet-anomaly-detection.git
cd ethernet-anomaly-detection

# 使用辅助脚本安装依赖
chmod +x scripts/install_dependencies.sh
sudo ./scripts/install_dependencies.sh

# 以开发模式安装
pip install -e .
### 手动安装
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装生产依赖
pip install -r requirements.txt

# 安装开发依赖（可选）
pip install -r requirements_dev.txt

# 安装项目
pip install -e .
## 使用方法

### 基本操作
# 启动系统
anomaly-detection start

# 查看系统状态
anomaly-detection status

# 停止系统
anomaly-detection stop
### 高级选项
# 指定网络接口
anomaly-detection start -i eth1

# 使用特定模型
anomaly-detection start --model lstm

# 查看帮助
anomaly-detection --help
### 模型管理
# 查看可用模型
anomaly-detection models list

# 训练新模型
anomaly-detection train once

# 切换模型版本
anomaly-detection models use --type xgboost --version 20230615_1230
## 配置说明

系统配置文件位于`config/`目录，主要配置文件包括：

- `config.yaml`：主配置文件（网络、告警等）
- `model_config.yaml`：模型参数配置
- `detection_rules.yaml`：规则引擎配置

可通过以下命令查看当前配置：anomaly-detection config show
## 告警处理

系统支持多种告警方式：
- 日志记录（默认）
- 邮件通知
- Slack集成

查看告警历史：anomaly-detection alerts list
## 开发指南

### 运行测试
# 运行所有测试
pytest

# 运行特定测试
pytest tests/unit/test_detection.py

# 生成测试覆盖率报告
pytest --cov=src tests/
### 代码规范

项目遵循PEP 8规范，使用以下工具确保代码质量：
- `black`：代码格式化
- `flake8`：代码检查
- `isort`：导入排序

提交代码前请运行：pre-commit run --all-files
## 许可证

本项目采用MIT许可证，详情参见[LICENSE](LICENSE)文件。

## 联系方式

- 项目主页：https://github.com/zdm65477730/ethernet-anomaly-detection
- 问题反馈：https://github.com/zdm65477730/ethernet-anomaly-detection/issues
- 技术支持：zdm65477730@126.com




这些文件构成了异常检测系统项目的基础配置，提供了完整的开发和生产环境支持：

1. **依赖管理**：
   - `requirements.txt` 包含生产环境必需的依赖，如网络抓包、数据处理和机器学习库
   - `requirements_dev.txt` 扩展了开发所需工具，包括测试框架、代码质量检查和文档生成工具

2. **项目安装配置**：
   - `setup.py` 定义了项目元数据、依赖关系和命令行入口，支持 `pip install -e .` 开发模式安装
   - 提供了分类的额外依赖（开发环境和GPU支持）

3. **项目文档**：
   - `README.md` 详细介绍了项目功能、安装步骤、使用方法和开发指南
   - 包含了徽章、架构说明和命令示例，便于新用户快速上手

4. **法律信息**：
   - `LICENSE` 使用MIT许可证，允许自由使用、修改和分发

5. **版本控制**：
   - `.gitignore` 精心配置了需要忽略的文件类型，包括虚拟环境、日志、数据和敏感配置

这些文件遵循了Python项目的最佳实践，使项目结构清晰、易于安装和维护，同时为开发团队提供了一致的工作环境配置。
