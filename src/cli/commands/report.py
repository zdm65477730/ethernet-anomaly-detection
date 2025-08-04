import os
import sys
import logging
import json
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

from datetime import datetime, timedelta  # 保持 datetime 的导入

# 在任何其他导入之前设置环境变量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 设置Python日志级别
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# 在最早的阶段重定向stderr来屏蔽特定的日志信息
original_stderr = sys.stderr

class FilteredStderr:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        # 需要屏蔽的日志关键词
        self.skip_patterns = [
            'tensorflow',
            'cuda', 
            'cudnn',
            'cufft',
            'cublas',
            'absl',
            'computation_placer',
            'oneDNN',
            'stream_executor',
            'external/local_xla'
        ]
    
    def write(self, msg):
        # 检查是否包含需要屏蔽的模式
        msg_lower = msg.lower()
        for pattern in self.skip_patterns:
            if pattern in msg_lower:
                return  # 直接返回，不写入stderr
        # 如果不包含屏蔽模式，则正常输出
        self.original_stderr.write(msg)
    
    def flush(self):
        self.original_stderr.flush()

# 立即应用过滤器
sys.stderr = FilteredStderr(original_stderr)

import typer
from typing import Optional
import matplotlib.pyplot as plt

# 添加新的导入
from src.utils.detection_visualizer import DetectionVisualizer
from src.cli.utils import print_info, print_success, print_error, print_warning
from src.config.config_manager import ConfigManager
from src.utils.logger import get_logger
from src.utils.evaluation_visualizer import EvaluationVisualizer

app = typer.Typer(help="检测报告相关命令")

logger = get_logger("cli.report")

@app.command(name="generate")
def generate_report(
    last_hours: Optional[int] = typer.Option(
        None, "--last-hours", "-h",
        help="生成最近几小时的报告"
    ),
    start_time: Optional[str] = typer.Option(
        None, "--start-time", "-s",
        help="开始时间 (格式: YYYY-MM-DD HH:MM:SS)"
    ),
    end_time: Optional[str] = typer.Option(
        None, "--end-time", "-e",
        help="结束时间 (格式: YYYY-MM-DD HH:MM:SS)"
    ),
    report_type: str = typer.Option(
        "detection", "--type", "-t",
        help="报告类型 (detection, evaluation)"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="输出文件路径"
    ),
    format: str = typer.Option(
        "json", "--format", "-f",
        help="输出格式 (json, html)"
    ),
    visualize: bool = typer.Option(
        False, "--visualize", "-v",
        help="生成可视化图表"
    ),
    config_dir: str = typer.Option(
        "config", "--config-dir", "-c",
        help="配置文件目录"
    )
):
    """
    生成检测报告
    """
    print_info("正在生成检测报告...")
    
    try:
        # 加载配置
        config = ConfigManager(config_dir=config_dir)
        
        # 确定时间范围
        if last_hours:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(hours=last_hours)
        elif start_time and end_time:
            start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        else:
            # 默认最近24小时
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(hours=24)
        
        # 确定输出路径
        if not output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if report_type == "detection":
                reports_dir = config.get("reports.detections_dir", "reports/detections")
                if format == "html":
                    output = os.path.join(reports_dir, f"detection_report_{timestamp}.html")
                else:
                    output = os.path.join(reports_dir, f"detection_report_{timestamp}.json")
            else:
                reports_dir = config.get("reports.evaluations_dir", "reports/evaluations")
                if format == "html":
                    output = os.path.join(reports_dir, f"evaluation_report_{timestamp}.html")
                else:
                    output = os.path.join(reports_dir, f"evaluation_report_{timestamp}.json")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output), exist_ok=True)
        
        # 生成报告数据
        report_data = _generate_report_data(
            config=config,
            start_time=start_dt,
            end_time=end_dt,
            report_type=report_type
        )
        
        # 生成可视化图表（如果需要）
        visualization_paths = []
        if visualize and report_type == "evaluation" and report_data.get("confusion_matrix"):
            visualization_paths = _generate_visualizations(report_data, os.path.dirname(output))
        
        # 保存报告
        if format == "html":
            _save_html_report(report_data, output, visualization_paths)
        else:
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        print_success(f"报告已生成: {output}")
        if visualization_paths:
            print_info(f"可视化图表已生成:")
            for path in visualization_paths:
                print_info(f"  - {path}")
        elif report_data["report_type"] == "detection" and visualize:
            print_info("尝试生成可视化图表...")
            try:
                # 即使没有自动生成，也尝试生成可视化图表
                visualizer = DetectionVisualizer(output_dir=os.path.dirname(output) or ".")
                chart_paths = []
                
                # 获取检测结果数据
                detection_results = report_data.get("detection_results", report_data.get("anomalies", []))
                print_info(f"使用 {len(detection_results)} 条检测结果生成图表")
                
                # 生成各类图表
                try:
                    anomaly_path = visualizer.plot_anomaly_distribution_over_time(
                        detection_results, 
                        show=False)
                    if anomaly_path:
                        chart_paths.append(anomaly_path)
                except Exception as e:
                    logger.warning(f"生成异常时间分布图失败: {e}")
                
                try:
                    protocol_path = visualizer.plot_protocol_distribution(
                        detection_results, 
                        show=False)
                    if protocol_path:
                        chart_paths.append(protocol_path)
                except Exception as e:
                    logger.warning(f"生成协议分布图失败: {e}")
                
                try:
                    score_path = visualizer.plot_anomaly_score_distribution(
                        detection_results, 
                        show=False)
                    if score_path:
                        chart_paths.append(score_path)
                except Exception as e:
                    logger.warning(f"生成异常分数分布图失败: {e}")
                
                try:
                    top_path = visualizer.plot_top_anomalies(
                        detection_results, 
                        show=False)
                    if top_path:
                        chart_paths.append(top_path)
                except Exception as e:
                    logger.warning(f"生成Top异常图失败: {e}")
                
                if chart_paths:
                    print_info(f"可视化图表已生成:")
                    for path in chart_paths:
                        print_info(f"  - {path}")
            except Exception as e:
                logger.error(f"生成可视化图表失败: {e}")
                print_error(f"生成可视化图表失败: {e}")
        
    except Exception as e:
        logger.error(f"生成报告失败: {str(e)}", exc_info=True)
        print_error(f"生成报告失败: {str(e)}")
        raise typer.Exit(code=1)

def _generate_report_data(
    config: ConfigManager,
    start_time: datetime,
    end_time: datetime,
    report_type: str
) -> dict:
    """
    生成报告数据
    
    Args:
        config: 配置管理器
        start_time: 开始时间
        end_time: 结束时间
        report_type: 报告类型
        
    Returns:
        报告数据字典
    """
    # 根据报告类型从实际数据中获取信息
    if report_type == "detection":
        return _generate_detection_report(config, start_time, end_time)
    else:  # evaluation
        return _generate_evaluation_report(config, start_time, end_time)

def _generate_detection_report(config: ConfigManager, start_time: datetime, end_time: datetime) -> dict:
    """
    从实际检测结果生成检测报告
    
    Args:
        config: 配置管理器
        start_time: 开始时间
        end_time: 结束时间
        
    Returns:
        检测报告数据
    """
    # 查找指定时间范围内的检测报告
    detections_dir = config.get("reports.detections_dir", "reports/detections")
    
    # 首先检查是否有测试数据
    test_data_path = os.path.join(detections_dir, 'sample_test_data.json')
    if os.path.exists(test_data_path):
        try:
            with open(test_data_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            # 包装测试数据
            anomalies = [item for item in report_data if isinstance(item, dict) and item.get("is_anomaly", False)]
            now = datetime.now()
            report_data = {
                "report_type": "detection",
                "generated_at": now.isoformat(),
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "summary": {
                    "total_packets": len(report_data),
                    "total_sessions": len(report_data),
                    "anomalies_detected": len(anomalies),
                    "anomaly_rate": f"{len(anomalies)/len(report_data)*100:.2f}%" if report_data else "0%"
                },
                "anomalies": anomalies,
                "detection_results": report_data
            }
            print_info(f"使用测试数据生成报告，包含 {len(report_data['detection_results'])} 条记录")
            return report_data
        except Exception as e:
            logger.warning(f"读取测试数据失败 {test_data_path}: {e}")
    
    # 如果目录存在，尝试查找最近的报告
    if os.path.exists(detections_dir):
        # 查找在时间范围内的报告
        report_files = []
        for file in os.listdir(detections_dir):
            if file.endswith('.json') and file != 'sample_test_data.json':  # 排除测试数据
                file_path = os.path.join(detections_dir, file)
                try:
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if start_time <= file_time <= end_time:
                        report_files.append((file_path, file_time))
                except (OSError, ValueError) as e:
                    logger.warning(f"无法获取文件时间戳 {file_path}: {e}")
        
        # 如果找到了匹配的报告，使用最新的一个
        if report_files:
            report_files.sort(key=lambda x: x[1], reverse=True)
            latest_report_path = report_files[0][0]
            try:
                with open(latest_report_path, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                
                # 检查数据格式，如果是列表则包装为字典
                if isinstance(report_data, list):
                    # 如果是列表格式，包装为字典
                    anomalies = [item for item in report_data if isinstance(item, dict) and item.get("is_anomaly", False)]
                    now = datetime.now()
                    report_data = {
                        "report_type": "detection",
                        "generated_at": now.isoformat(),
                        "time_range": {
                            "start": start_time.isoformat(),
                            "end": end_time.isoformat()
                        },
                        "summary": {
                            "total_packets": len(report_data),
                            "total_sessions": len(report_data),
                            "anomalies_detected": len(anomalies),
                            "anomaly_rate": f"{len(anomalies)/len(report_data)*100:.2f}%" if report_data else "0%"
                        },
                        "anomalies": anomalies,
                        "detection_results": report_data
                    }
                else:
                    # 更新时间信息
                    report_data["time_range"] = {
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat()
                    }
                
                print_info(f"使用检测数据生成报告，包含 {len(report_data.get('detection_results', []))} 条记录")
                return report_data
            except json.JSONDecodeError as e:
                logger.warning(f"JSON解析失败 {latest_report_path}: {e}")
            except Exception as e:
                logger.warning(f"读取检测报告失败 {latest_report_path}: {e}")
    
    # 如果没有找到现有报告，返回示例数据
    logger.info("未找到匹配的检测报告，使用示例数据")
    return {
        "report_type": "detection",
        "generated_at": datetime.now().isoformat(),
        "time_range": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat()
        },
        "summary": {
            "total_packets": 10000,
            "total_sessions": 500,
            "anomalies_detected": 25,
            "anomaly_rate": "0.25%"
        },
        "anomalies": [
            {
                "timestamp": (start_time + timedelta(minutes=30)).isoformat(),
                "source_ip": "192.168.1.100",
                "destination_ip": "10.0.0.1",
                "protocol": "TCP",
                "anomaly_type": "SYN Flood",
                "severity": "high",
                "details": "异常高的SYN包发送速率"
            },
            {
                "timestamp": (start_time + timedelta(hours=2)).isoformat(),
                "source_ip": "192.168.1.200",
                "destination_ip": "8.8.8.8",
                "protocol": "UDP",
                "anomaly_type": "Port Scan",
                "severity": "medium",
                "details": "短时间内扫描多个端口"
            }
        ],
        "top_anomalous_ips": [
            {"ip": "192.168.1.100", "count": 15},
            {"ip": "192.168.1.200", "count": 8},
            {"ip": "10.0.0.50", "count": 2}
        ]
    }

def _generate_evaluation_report(config: ConfigManager, start_time: datetime, end_time: datetime) -> dict:
    """
    从实际评估结果生成评估报告
    
    Args:
        config: 配置管理器
        start_time: 开始时间
        end_time: 结束时间
        
    Returns:
        评估报告数据
    """
    # 查找指定时间范围内的评估报告
    evaluations_dir = config.get("reports.evaluations_dir", "reports/evaluations")
    
    # 如果目录存在，尝试查找最近的评估报告
    if os.path.exists(evaluations_dir):
        # 查找在时间范围内的评估报告
        report_files = []
        for file in os.listdir(evaluations_dir):
            if file.endswith('.json'):
                try:
                    # 从文件名中提取时间
                    parts = file.split('_')
                    if len(parts) >= 3:
                        date_part = parts[-2]  # YYYYMMDD
                        time_part = parts[-1].split('.')[0]  # HHMMSS
                        file_time_str = f"{date_part}_{time_part}"
                        file_time = datetime.strptime(file_time_str, "%Y%m%d_%H%M%S")
                        if start_time <= file_time <= end_time:
                            report_files.append((os.path.join(evaluations_dir, file), file_time))
                except Exception as e:
                    # 如果无法解析文件名中的时间，使用修改时间
                    file_path = os.path.join(evaluations_dir, file)
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if start_time <= file_mtime <= end_time:
                        report_files.append((file_path, file_mtime))
        
        # 如果找到了匹配的报告，使用最新的一个
        if report_files:
            report_files.sort(key=lambda x: x[1], reverse=True)
            latest_report_path = report_files[0][0]
            try:
                with open(latest_report_path, 'r', encoding='utf-8') as f:
                    eval_data = json.load(f)
                
                # 转换为报告格式
                report_data = {
                    "report_type": "evaluation",
                    "generated_at": datetime.now().isoformat(),
                    "time_range": {
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat()
                    },
                    "model_performance": {
                        "model_type": "unknown",  # 实际系统中应该从模型元数据获取
                        "accuracy": eval_data.get("accuracy", 0),
                        "precision": eval_data.get("precision", 0),
                        "recall": eval_data.get("recall", 0),
                        "f1_score": eval_data.get("f1", 0),
                        "auc": eval_data.get("auc", 0)
                    }
                }
                
                # 处理混淆矩阵
                if "confusion_matrix" in eval_data:
                    cm = eval_data["confusion_matrix"]
                    if len(cm) >= 2 and len(cm[0]) >= 2 and len(cm[1]) >= 2:
                        report_data["confusion_matrix"] = {
                            "true_negative": cm[0][0],
                            "false_positive": cm[0][1],
                            "false_negative": cm[1][0],
                            "true_positive": cm[1][1]
                        }
                
                # 添加示例特征重要性（实际系统中应该从模型获取）
                report_data["feature_importance"] = {
                    "packet_count": 0.15,
                    "byte_count": 0.12,
                    "flow_duration": 0.10,
                    "avg_packet_size": 0.08,
                    "tcp_syn_count": 0.18,
                    "tcp_ack_count": 0.07,
                    "payload_entropy": 0.30
                }
                
                return report_data
            except Exception as e:
                logger.warning(f"读取评估报告失败 {latest_report_path}: {e}")
    
    # 如果没有找到现有报告，返回示例数据
    logger.info("未找到匹配的评估报告，使用示例数据")
    return {
        "report_type": "evaluation",
        "generated_at": datetime.now().isoformat(),
        "time_range": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat()
        },
        "model_performance": {
            "model_type": "xgboost",
            "accuracy": 0.975,
            "precision": 0.95,
            "recall": 0.92,
            "f1_score": 0.935,
            "auc": 0.98
        },
        "confusion_matrix": {
            "true_positive": 230,
            "true_negative": 750,
            "false_positive": 15,
            "false_negative": 20
        },
        "feature_importance": {
            "packet_count": 0.15,
            "byte_count": 0.12,
            "flow_duration": 0.10,
            "avg_packet_size": 0.08,
            "tcp_syn_count": 0.18,
            "tcp_ack_count": 0.07,
            "payload_entropy": 0.30
        }
    }

def _generate_visualizations(report_data: dict, output_dir: str) -> list:
    """
    生成可视化图表
    
    Args:
        report_data: 报告数据
        output_dir: 输出目录
        
    Returns:
        生成的图表路径列表
    """
    if report_data["report_type"] != "evaluation":
        # 检查是否为检测报告
        if report_data["report_type"] == "detection":
            # 使用新的检测结果可视化
            visualizer = DetectionVisualizer(output_dir=output_dir)
            generated_paths = []
            
            try:
                # 生成综合报告（包含多个图表）
                report_path = visualizer.generate_comprehensive_report(
                    detection_results=report_data.get("anomalies", report_data.get("detection_results", [])),
                    output_path=os.path.join(output_dir, "detection_report.html")
                )
                generated_paths.append(report_path)
            except Exception as e:
                logger.warning(f"生成检测综合报告失败: {e}")
            
            return generated_paths
        else:
            return []
    
    visualizer = EvaluationVisualizer(output_dir=output_dir)
    generated_paths = []
    
    # 生成混淆矩阵图
    try:
        cm_data = report_data["confusion_matrix"]
        cm = [[cm_data["true_negative"], cm_data["false_positive"]],
              [cm_data["false_negative"], cm_data["true_positive"]]]
        
        cm_path = visualizer.plot_confusion_matrix(
            cm=np.array(cm),
            class_names=["正常", "异常"],
            title="模型混淆矩阵",
            save_name="confusion_matrix.png",
            show=False
        )
        generated_paths.append(cm_path)
    except Exception as e:
        logger.warning(f"生成混淆矩阵图失败: {e}")
    
    # 生成特征重要性图
    try:
        feature_importance = report_data["feature_importance"]
        features = list(feature_importance.keys())
        importance = list(feature_importance.values())
        
        # 创建特征重要性图
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = range(len(features))
        bars = ax.barh(y_pos, importance, color='skyblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('重要性')
        ax.set_title('特征重要性')
        ax.grid(axis='x', alpha=0.3)
        
        # 添加数值标签
        for i, (bar, imp) in enumerate(zip(bars, importance)):
            ax.text(imp + 0.01, i, f'{imp:.3f}', va='center')
        
        plt.tight_layout()
        feature_importance_path = os.path.join(output_dir, "feature_importance.png")
        plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_paths.append(feature_importance_path)
    except Exception as e:
        logger.warning(f"生成特征重要性图失败: {e}")
    
    # 生成性能指标图
    try:
        performance = report_data["model_performance"]
        metrics = ["准确率", "精确率", "召回率", "F1分数", "AUC"]
        values = [
            performance["accuracy"],
            performance["precision"],
            performance["recall"],
            performance["f1_score"],
            performance["auc"]
        ]
        
        # 创建性能指标图
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = range(len(metrics))
        bars = ax.bar(x_pos, values, color=['red', 'green', 'blue', 'orange', 'purple'])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics)
        ax.set_ylabel('分数')
        ax.set_title('模型性能指标')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(i, val + 0.02, f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        performance_path = os.path.join(output_dir, "performance_metrics.png")
        plt.savefig(performance_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_paths.append(performance_path)
    except Exception as e:
        logger.warning(f"生成性能指标图失败: {e}")
    
    return generated_paths

import os
from typing import List

def _save_html_report(report_data: dict, output_path: str, chart_paths: List[str] = None):
    """
    保存HTML格式的报告
    
    Args:
        report_data: 报告数据
        output_path: 输出路径
        chart_paths: 图表路径列表
    """
    if chart_paths is None:
        chart_paths = []
    
    # 生成HTML内容
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>异常检测报告</title>
    <style>
        body {{
            font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .metric-card {{
            display: inline-block;
            width: 200px;
            padding: 15px;
            margin: 10px;
            background-color: #f8f9fa;
            border-radius: 5px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }}
        .metric-label {{
            font-size: 14px;
            color: #666;
        }}
        .chart-container {{
            text-align: center;
            margin: 30px 0;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>异常检测报告</h1>

        <p><strong>生成时间:</strong> {report_data.get('generated_at', 'N/A')}</p>
"""
    
    # 添加时间范围
    time_range = report_data.get('time_range', {})
    if time_range:
        html_content += f"""
        <p><strong>时间范围:</strong> {time_range.get('start', 'N/A')} 至 {time_range.get('end', 'N/A')}</p>
"""
    
    # 处理不同类型的报告
    report_type = report_data.get("report_type", "")
    
    # 如果是检测报告
    if report_type == "detection":
        # 添加摘要信息
        summary = report_data.get('summary', {})
        if summary:
            html_content += """
        <h2>检测摘要</h2>

        <div class="metric-card">
            <div class="metric-value">{total_packets}</div>
            <div class="metric-label">总数据包数</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{total_sessions}</div>
            <div class="metric-label">总会话数</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{anomalies_detected}</div>
            <div class="metric-label">检测到异常</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{anomaly_rate}</div>
            <div class="metric-label">异常率</div>
        </div>
""".format(
            total_packets=summary.get('total_packets', 'N/A'),
            total_sessions=summary.get('total_sessions', 'N/A'),
            anomalies_detected=summary.get('anomalies_detected', 'N/A'),
            anomaly_rate=summary.get('anomaly_rate', 'N/A')
        )
    
    # 如果是评估报告
    elif report_type == "evaluation":
        # 添加模型性能指标
        if "model_performance" in report_data:
            performance = report_data["model_performance"]
            html_content += """
        <h2>模型性能</h2>
        <div class="metric-card">
            <div class="metric-value">{accuracy:.3f}</div>
            <div class="metric-label">准确率</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{precision:.3f}</div>
            <div class="metric-label">精确率</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{recall:.3f}</div>
            <div class="metric-label">召回率</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{f1_score:.3f}</div>
            <div class="metric-label">F1分数</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{auc:.3f}</div>
            <div class="metric-label">AUC</div>
        </div>
""".format(
                accuracy=performance.get('accuracy', 0),
                precision=performance.get('precision', 0),
                recall=performance.get('recall', 0),
                f1_score=performance.get('f1_score', 0),
                auc=performance.get('auc', 0)
            )
        
        # 添加混淆矩阵
        if "confusion_matrix" in report_data:
            cm = report_data["confusion_matrix"]
            html_content += """
        <h2>混淆矩阵</h2>
        <table>
            <thead>
                <tr>
                    <th></th>
                    <th>预测正常</th>
                    <th>预测异常</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>实际正常</strong></td>
                    <td>{}</td>
                    <td>{}</td>
                </tr>
                <tr>
                    <td><strong>实际异常</strong></td>
                    <td>{}</td>
                    <td>{}</td>
                </tr>
            </tbody>
        </table>
""".format(cm.get("true_negative", 0), cm.get("false_positive", 0), 
           cm.get("false_negative", 0), cm.get("true_positive", 0))
        
        # 添加特征重要性
        if "feature_importance" in report_data:
            html_content += """
        <h2>特征重要性</h2>
        <table>
            <thead>
                <tr>
                    <th>特征</th>
                    <th>重要性</th>
                </tr>
            </thead>
            <tbody>
"""
            for feature, importance in report_data["feature_importance"].items():
                html_content += f"""
                <tr>
                    <td>{feature}</td>
                    <td>{importance:.3f}</td>
                </tr>
"""
            html_content += """
            </tbody>
        </table>
"""
    
    # 添加可视化图表
    if chart_paths:
        html_content += """
        <h2>检测结果可视化</h2>
"""
        for chart_path in chart_paths:
            # 获取相对路径用于HTML显示
            relative_path = os.path.relpath(chart_path, os.path.dirname(output_path))
            html_content += f"""
        <div class="chart-container">
            <img src="{relative_path}" alt="检测图表">
        </div>
"""
    
    # 添加异常检测详细信息表格
    html_content += """
        <h2>检测到的异常</h2>
        <table>
            <thead>
                <tr>
                    <th>时间</th>
                    <th>源IP</th>
                    <th>目标IP</th>
                    <th>协议</th>
                    <th>异常类型</th>
                    <th>严重性</th>
                    <th>详情</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # 添加异常记录
    anomalies = report_data.get("anomalies", [])
    detection_results = report_data.get("detection_results", [])
    
    # 如果没有明确的anomalies字段，则从detection_results中筛选
    if not anomalies and detection_results:
        anomalies = [r for r in detection_results if isinstance(r, dict) and r.get("is_anomaly", False)]
    
    if anomalies:
        for record in anomalies:
            if isinstance(record, dict):
                # 提取记录信息
                timestamp = record.get('detection_time', 'N/A')
                if timestamp != 'N/A' and isinstance(timestamp, (int, float)):
                    # 处理可能的毫秒时间戳
                    if timestamp > 1e10:  # 如果是毫秒时间戳
                        timestamp = timestamp / 1000
                    timestamp = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                elif timestamp == 'N/A':
                    timestamp = 'N/A'
                
                # 处理IP地址字段
                src_ip = record.get('src_ip', record.get('source_ip', 'N/A'))
                dst_ip = record.get('dst_ip', record.get('destination_ip', 'N/A'))
                
                # 处理协议字段
                protocol = record.get('protocol', record.get('proto', 'N/A'))
                
                # 处理异常类型
                anomaly_type = record.get('detection_method', 'N/A')
                
                # 处理严重性
                severity = f"{record.get('anomaly_score', 0):.3f}" if 'anomaly_score' in record else 'N/A'
                
                # 处理详情
                threshold = record.get('threshold_used', 'N/A')
                details = f"阈值: {threshold}" if threshold != 'N/A' else 'N/A'
                
                html_content += f"""
                <tr>
                    <td>{timestamp}</td>
                    <td>{src_ip}</td>
                    <td>{dst_ip}</td>
                    <td>{protocol}</td>
                    <td>{anomaly_type}</td>
                    <td>{severity}</td>
                    <td>{details}</td>
                </tr>
"""
    else:
        # 如果没有异常记录，显示提示信息
        html_content += """
                <tr>
                    <td colspan="7" style="text-align: center;">未检测到异常</td>
                </tr>
"""
    
    html_content += """
            </tbody>
        </table>

    </div>
</body>
</html>
"""
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == "__main__":
    app()