import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import json
from datetime import datetime
import logging

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'STHeiti', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 禁用字体警告
import warnings
warnings.filterwarnings('ignore', module='matplotlib')
# 特别忽略字体相关的警告
warnings.filterwarnings('ignore', message='.*Glyph.*missing from font.*')

# 设置matplotlib后端
plt.switch_backend('Agg')


class DetectionVisualizer:
    """检测结果可视化工具，生成异常分布、协议分布等检测图表"""
    
    def __init__(self, output_dir: str = "detection_plots"):
        """
        初始化可视化工具
        
        参数:
            output_dir: 图表输出目录
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置默认样式
        sns.set_style("whitegrid")
        plt.style.use("seaborn-v0_8-colorblind")
    
    def plot_anomaly_distribution_over_time(
        self,
        detection_results: List[Dict],
        time_interval: str = "1h",
        save_name: Optional[str] = None,
        show: bool = True
    ) -> str:
        """
        绘制异常随时间分布的图表
        
        参数:
            detection_results: 检测结果列表
            time_interval: 时间间隔（如"1H"表示1小时，"1D"表示1天）
            save_name: 保存文件名，为None则自动生成
            show: 是否显示图表
            
        返回:
            保存文件路径
        """
        # 转换为DataFrame
        df = pd.DataFrame(detection_results)
        
        # 确保有时间戳列
        if 'timestamp' not in df.columns and 'detection_time' not in df.columns:
            raise ValueError("检测结果中缺少时间戳字段")
            
        time_col = 'timestamp' if 'timestamp' in df.columns else 'detection_time'
        # 处理时间戳格式
        try:
            df[time_col] = pd.to_datetime(df[time_col], unit='s' if isinstance(df[time_col].iloc[0], (int, float)) else None)
        except Exception:
            # 如果转换失败，尝试其他方式
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        
        # 移除无效的时间戳
        df = df.dropna(subset=[time_col])
        
        # 检查是否有足够的数据
        if df.empty:
            raise ValueError("没有有效的时间戳数据")
        
        # 检查是否有is_anomaly列
        if 'is_anomaly' not in df.columns:
            raise ValueError("检测结果中缺少is_anomaly字段")
        
        # 设置时间索引
        df.set_index(time_col, inplace=True)
        
        # 按时间间隔重采样
        try:
            anomaly_counts = df[df['is_anomaly'] == True].resample(time_interval).size()
            normal_counts = df[df['is_anomaly'] == False].resample(time_interval).size()
        except Exception as e:
            raise ValueError(f"重采样失败: {e}")
        
        # 绘制图表
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(anomaly_counts.index, anomaly_counts.values, label='异常', marker='o', linewidth=2)
        ax.plot(normal_counts.index, normal_counts.values, label='正常', marker='s', linewidth=2)
        
        ax.set_xlabel('时间')
        ax.set_ylabel('数量')
        ax.set_title('异常随时间分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存图表
        if save_name is None:
            save_name = f"anomaly_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        
        return save_path
    
    def plot_protocol_distribution(
        self,
        detection_results: List[Dict],
        save_name: Optional[str] = None,
        show: bool = True
    ) -> str:
        """
        绘制协议分布饼图
        
        参数:
            detection_results: 检测结果列表
            save_name: 保存文件名，为None则自动生成
            show: 是否显示图表
            
        返回:
            保存文件路径
        """
        # 转换为DataFrame
        df = pd.DataFrame(detection_results)
        
        # 检查协议列
        protocol_columns = ['protocol', 'proto']
        protocol_col = None
        for col in protocol_columns:
            if col in df.columns:
                protocol_col = col
                break
        
        # 如果没有协议字段，不抛出异常，而是返回None表示跳过该图表
        if protocol_col is None:
            print("提示: 检测结果中缺少协议字段，跳过协议分布图生成")
            return None
        
        # 统计协议分布
        protocol_counts = df[protocol_col].value_counts()
        
        # 如果没有数据，返回None
        if protocol_counts.empty:
            print("提示: 没有协议数据可用于生成分布图")
            return None
        
        # 绘制饼图
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(protocol_counts)))
        wedges, texts, autotexts = ax.pie(
            protocol_counts.values, 
            labels=protocol_counts.index, 
            autopct=lambda pct: f'{pct:.1f}%' if pct > 5 else '',  # 只显示占比大于5%的标签
            colors=colors,
            startangle=90
        )
        
        # 添加图例，显示所有协议及其数量
        legend_labels = [f'{idx}: {count}' for idx, count in protocol_counts.items()]
        ax.legend(wedges, legend_labels, title="协议分布", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        ax.set_title('协议分布')
        
        # 美化文本
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        # 保存图表
        if save_name is None:
            save_name = f"protocol_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        
        return save_path
    
    def plot_anomaly_score_distribution(
        self,
        detection_results: List[Dict],
        bins: int = 50,
        save_name: Optional[str] = None,
        show: bool = True
    ) -> str:
        """
        绘制异常分数分布直方图
        
        参数:
            detection_results: 检测结果列表
            bins: 直方图bins数量
            save_name: 保存文件名，为None则自动生成
            show: 是否显示图表
            
        返回:
            保存文件路径
        """
        # 转换为DataFrame
        df = pd.DataFrame(detection_results)
        
        # 检查异常分数列
        if 'anomaly_score' not in df.columns:
            raise ValueError("检测结果中缺少异常分数字段")
        
        # 移除无效的分数
        df = df.dropna(subset=['anomaly_score'])
        df['anomaly_score'] = pd.to_numeric(df['anomaly_score'], errors='coerce')
        df = df.dropna(subset=['anomaly_score'])
        
        if df.empty:
            raise ValueError("没有有效的异常分数数据")
        
        # 绘制直方图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 分别绘制正常和异常的分数分布
        normal_scores = df[df['is_anomaly'] == False]['anomaly_score'] if 'is_anomaly' in df.columns else df['anomaly_score']
        anomaly_scores = df[df['is_anomaly'] == True]['anomaly_score'] if 'is_anomaly' in df.columns else pd.Series(dtype=float)
        
        ax.hist(normal_scores, bins=bins, alpha=0.7, label='正常' if 'is_anomaly' in df.columns else '分数分布', 
                color='blue', edgecolor='black', linewidth=0.5)
        if not anomaly_scores.empty:
            ax.hist(anomaly_scores, bins=bins, alpha=0.7, label='异常', color='red', edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('异常分数')
        ax.set_ylabel('频次')
        ax.set_title('异常分数分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        if save_name is None:
            save_name = f"anomaly_score_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        
        return save_path
    
    def plot_top_anomalies(
        self,
        detection_results: List[Dict],
        top_n: int = 10,
        save_name: Optional[str] = None,
        show: bool = True
    ) -> str:
        """
        绘制Top异常样本条形图
        
        参数:
            detection_results: 检测结果列表
            top_n: 显示Top N个异常
            save_name: 保存文件名，为None则自动生成
            show: 是否显示图表
            
        返回:
            保存文件路径
        """
        # 转换为DataFrame
        df = pd.DataFrame(detection_results)
        
        # 检查异常分数列
        if 'anomaly_score' not in df.columns:
            raise ValueError("检测结果中缺少异常分数字段")
        
        # 移除无效的分数
        df = df.dropna(subset=['anomaly_score'])
        df['anomaly_score'] = pd.to_numeric(df['anomaly_score'], errors='coerce')
        df = df.dropna(subset=['anomaly_score'])
        
        if df.empty:
            raise ValueError("没有有效的异常分数数据")
        
        # 获取Top异常
        top_anomalies = df.nlargest(top_n, 'anomaly_score')
        
        # 绘制条形图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = range(len(top_anomalies))
        scores = top_anomalies['anomaly_score'].values
        
        bars = ax.barh(y_pos, scores, color='red', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # 设置y轴标签
        if 'session_id' in top_anomalies.columns:
            labels = [f"Session {sid[:8]}" for sid in top_anomalies['session_id']]
        elif 'src_ip' in top_anomalies.columns and 'dst_ip' in top_anomalies.columns:
            labels = [f"{src} -> {dst}" for src, dst in zip(top_anomalies['src_ip'], top_anomalies['dst_ip'])]
        else:
            labels = [f"Sample {i}" for i in range(len(top_anomalies))]
            
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('异常分数')
        ax.set_title(f'Top {top_n} 异常样本')
        ax.grid(axis='x', alpha=0.3)
        
        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 0.01, i, f'{score:.3f}', va='center')
        
        plt.tight_layout()
        
        # 保存图表
        if save_name is None:
            save_name = f"top_anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        
        return save_path
    
    def generate_comprehensive_report(
        self,
        detection_results: List[Dict],
        output_path: str = None
    ) -> str:
        """
        生成综合检测报告（包含多个图表）
        
        参数:
            detection_results: 检测结果列表
            output_path: 输出HTML文件路径
            
        返回:
            生成的HTML文件路径
        """
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 生成所有图表
        chart_paths = []
        
        try:
            anomaly_dist_path = self.plot_anomaly_distribution_over_time(detection_results, show=False)
            if anomaly_dist_path:
                chart_paths.append(anomaly_dist_path)
        except Exception as e:
            print(f"生成异常时间分布图失败: {e}")
            
        try:
            protocol_dist_path = self.plot_protocol_distribution(detection_results, show=False)
            if protocol_dist_path:
                chart_paths.append(protocol_dist_path)
        except Exception as e:
            print(f"生成协议分布图失败: {e}")
            
        try:
            score_dist_path = self.plot_anomaly_score_distribution(detection_results, show=False)
            if score_dist_path:
                chart_paths.append(score_dist_path)
        except Exception as e:
            print(f"生成异常分数分布图失败: {e}")
            
        try:
            top_anomalies_path = self.plot_top_anomalies(detection_results, show=False)
            if top_anomalies_path:
                chart_paths.append(top_anomalies_path)
        except Exception as e:
            print(f"生成Top异常图失败: {e}")
        
        # 生成HTML报告
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        
        html_content = self._generate_html_report(detection_results, chart_paths)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return output_path
    
    def _generate_html_report(self, detection_results: List[Dict], chart_paths: List[str]) -> str:
        """生成HTML报告"""
        # 基本统计
        total_count = len(detection_results)
        anomaly_count = sum(1 for r in detection_results if isinstance(r, dict) and r.get('is_anomaly', False))
        anomaly_rate = anomaly_count / total_count if total_count > 0 else 0
        
        # 生成HTML内容
        html = f"""
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
        .chart {{
            text-align: center;
            margin: 30px 0;
        }}
        .chart img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            color: #666;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>异常检测报告</h1>
        
        <div class="summary">
            <div class="metric-card">
                <div class="metric-label">总样本数</div>
                <div class="metric-value">{total_count:,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">异常数量</div>
                <div class="metric-value">{anomaly_count:,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">异常率</div>
                <div class="metric-value">{anomaly_rate:.2%}</div>
            </div>
        </div>
        
        <h2>检测结果可视化</h2>
        """
        
        # 添加图表
        for chart_path in chart_paths:
            # 获取相对路径用于HTML显示
            relative_path = os.path.relpath(chart_path, os.path.dirname(chart_paths[0]) if chart_paths else self.output_dir)
            html += f"""
        <div class="chart">
            <img src="{relative_path}" alt="检测图表">
        </div>
            """
        
        # 添加异常检测详细信息表格
        html += """
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
        
        # 添加异常记录（最多显示50条）
        anomaly_records = [r for r in detection_results if isinstance(r, dict) and r.get('is_anomaly', False)]
        displayed_records = anomaly_records[:50] if anomaly_records else []
        
        if displayed_records:
            for record in displayed_records:
                if isinstance(record, dict):
                    # 提取记录信息
                    timestamp = record.get('detection_time', 'N/A')
                    if timestamp != 'N/A' and isinstance(timestamp, (int, float)):
                        from datetime import datetime
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
                    
                    html += f"""
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
            html += """
                <tr>
                    <td colspan="7" style="text-align: center;">未检测到异常</td>
                </tr>
            """
        
        html += """
            </tbody>
        </table>
        """
        
        # 添加正常检测样本（如果异常记录少于10条）
        if len(anomaly_records) < 10:
            normal_records = [r for r in detection_results if isinstance(r, dict) and not r.get('is_anomaly', True)]
            additional_records = normal_records[:10] if normal_records else []
            
            if additional_records:
                html += """
        <h2>正常检测样本示例</h2>
        <table>
            <thead>
                <tr>
                    <th>时间</th>
                    <th>源IP</th>
                    <th>目标IP</th>
                    <th>协议</th>
                    <th>异常分数</th>
                    <th>检测方法</th>
                    <th>详情</th>
                </tr>
            </thead>
            <tbody>
                """
                
                for record in additional_records:
                    if isinstance(record, dict):
                        # 提取记录信息
                        timestamp = record.get('detection_time', 'N/A')
                        if timestamp != 'N/A' and isinstance(timestamp, (int, float)):
                            from datetime import datetime
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
                        
                        # 处理异常分数
                        anomaly_score = f"{record.get('anomaly_score', 0):.3f}" if 'anomaly_score' in record else 'N/A'
                        
                        # 处理检测方法
                        detection_method = record.get('detection_method', 'N/A')
                        
                        # 处理详情
                        threshold = record.get('threshold_used', 'N/A')
                        details = f"阈值: {threshold}" if threshold != 'N/A' else 'N/A'
                        
                        html += f"""
                <tr>
                    <td>{timestamp}</td>
                    <td>{src_ip}</td>
                    <td>{dst_ip}</td>
                    <td>{protocol}</td>
                    <td>{anomaly_score}</td>
                    <td>{detection_method}</td>
                    <td>{details}</td>
                </tr>
                        """
                
                html += """
            </tbody>
        </table>
                """
        
        html += f"""
        <div class="footer">
            <p>报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>异常检测系统</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html
