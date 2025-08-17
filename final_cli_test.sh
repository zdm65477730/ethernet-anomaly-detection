#!/bin/bash

# 使用CLI命令完成异常检测系统完整工作流程测试脚本（最终版）
# 该脚本使用anomaly-detector CLI命令执行完整的端到端流程

set -e  # 遇到错误时停止执行

echo "========================================"
echo "  使用CLI命令完成异常检测系统完整流程测试（最终版）"
echo "========================================"

# 创建测试目录
TEST_DIR="/tmp/anomaly_detection_final_cli_test_$(date +%s)"
mkdir -p "$TEST_DIR"
echo "测试目录: $TEST_DIR"

# 创建必要的子目录
mkdir -p "$TEST_DIR/data"
mkdir -p "$TEST_DIR/models"
mkdir -p "$TEST_DIR/reports"

echo "已创建测试目录结构"

echo "========================================"
echo "  步骤1: 初始化系统"
echo "========================================"

# 初始化系统
anomaly-detector init

echo "系统初始化完成"

echo "========================================"
echo "  步骤2: 生成测试数据"
echo "========================================"

# 生成测试数据
anomaly-detector generate-test-data generate-test-data --count 500 --output "$TEST_DIR/data"

if [ -f "$TEST_DIR/data/model_features_data.csv" ]; then
    echo "✓ 数据生成成功"
else
    echo "✗ 数据生成失败"
    exit 1
fi

echo "========================================"
echo "  步骤3: 训练XGBoost模型"
echo "========================================"

# 训练XGBoost模型
anomaly-detector train --model xgboost \
  --data "$TEST_DIR/data/model_features_data.csv" \
  --output "$TEST_DIR/models" \
  --test-size 0.2

echo "XGBoost模型训练完成"

echo "========================================"
echo "  步骤4: 训练MLP模型"
echo "========================================"

# 训练MLP模型
anomaly-detector train --model mlp \
  --data "$TEST_DIR/data/model_features_data.csv" \
  --output "$TEST_DIR/models" \
  --test-size 0.2

echo "MLP模型训练完成"

echo "========================================"
echo "  步骤5: 启动自驱动学习系统"
echo "========================================"

echo "启动自驱动学习系统（将在后台运行5秒后停止）..."

# 启动自驱动学习系统（在后台运行）
timeout 5s anomaly-detector train self-driving || true

echo "自驱动学习系统测试完成"

echo "========================================"
echo "  步骤6: 生成检测报告"
echo "========================================"

# 生成JSON格式的检测报告
anomaly-detector report generate \
  --type detection \
  --format json \
  --output "$TEST_DIR/reports/detection_report.json"

echo "检测报告生成完成"

echo "========================================"
echo "  测试完成"
echo "========================================"

echo "使用CLI命令的完整测试流程总结:"
echo "1. 系统初始化 - 已完成"
echo "2. 数据生成 - 已完成"
echo "3. XGBoost模型训练 - 已完成"
echo "4. MLP模型训练 - 已完成"
echo "5. 自驱动学习系统测试 - 已完成"
echo "6. 检测报告生成 - 已完成"

echo ""
echo "测试数据保存在: $TEST_DIR"
echo "请查看 $TEST_DIR/reports 目录中的检测报告"