import os
import time
import json
import logging
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from src.system.base_component import BaseComponent
from src.utils.logger import get_logger
from src.config.config_manager import ConfigManager

class AlertManager(BaseComponent):
    """告警管理器，负责异常检测结果的日志记录、报告生成和告警发送"""
    
    def __init__(self, config=None):
        super().__init__()
        self.logger = get_logger("alert_manager")
        self.config = config or ConfigManager()
        
        # 初始化配置
        self.alert_threshold = self.config.get("alert.threshold", 0.7)  # 告警阈值
        self.alert_interval = self.config.get("alert.interval", 60)  # 告警间隔(秒)，避免风暴
        
        # 告警报告目录
        self.reports_dir = self.config.get("alert.reports_dir", "reports/anomalies")
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # 最近告警时间，用于控制告警频率
        self.last_alert_time = {}
        
        # 邮件配置
        self.smtp_config = {
            "enabled": self.config.get("smtp.enabled", False),
            "server": self.config.get("smtp.server", "smtp.example.com"),
            "port": self.config.get("smtp.port", 587),
            "use_tls": self.config.get("smtp.use_tls", True),
            "username": self.config.get("smtp.username", ""),
            "password": self.config.get("smtp.password", ""),
            "from_email": self.config.get("smtp.from_email", "alerts@example.com"),
            "to_emails": self.config.get("smtp.to_emails", ["admin@example.com"])
        }
        
        # 初始化邮件服务器连接
        self.smtp_server = None
        
    def start(self):
        """启动告警管理器"""
        if self.is_running:
            self.logger.warning("告警管理器已在运行中")
            return
            
        super().start()
        self.logger.info("告警管理器已启动")
    
    def stop(self):
        """停止告警管理器，清理资源"""
        if not self.is_running:
            return
            
        super().stop()
        
        # 关闭SMTP连接
        if hasattr(self, 'smtp_server') and self.smtp_server:
            try:
                self.smtp_server.quit()
                self.logger.info("SMTP服务器连接已关闭")
            except Exception as e:
                self.logger.error(f"关闭SMTP连接时出错: {str(e)}")
        
        self.logger.info("告警管理器已停止")
    
    def _init_smtp(self):
        """初始化SMTP服务器连接"""
        if not self.smtp_config["enabled"]:
            return False
            
        try:
            # 如果已有连接，先关闭
            if self.smtp_server:
                self.smtp_server.quit()
                
            # 创建新连接
            self.smtp_server = smtplib.SMTP(
                self.smtp_config["server"], 
                self.smtp_config["port"]
            )
            
            # 启用TLS
            if self.smtp_config["use_tls"]:
                self.smtp_server.starttls()
                
            # 登录
            if self.smtp_config["username"] and self.smtp_config["password"]:
                self.smtp_server.login(
                    self.smtp_config["username"], 
                    self.smtp_config["password"]
                )
                
            self.logger.info("SMTP服务器连接成功")
            return True
            
        except Exception as e:
            self.logger.error(f"SMTP服务器连接失败: {str(e)}")
            self.smtp_server = None
            return False
    
    def _send_email_alert(self, anomaly):
        """发送邮件告警"""
        if not self.smtp_config["enabled"]:
            return False
            
        # 确保SMTP连接有效
        if not self.smtp_server and not self._init_smtp():
            return False
            
        try:
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config["from_email"]
            msg['To'] = ", ".join(self.smtp_config["to_emails"])
            msg['Subject'] = f"网络异常告警: {anomaly.get('anomaly_type', '未知异常')}"
            
            # 邮件内容
            body = f"""
            检测到网络异常:
            
            时间: {datetime.fromtimestamp(anomaly['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}
            会话ID: {anomaly.get('session_id', '未知')}
            源IP: {anomaly.get('src_ip', '未知')}
            目的IP: {anomaly.get('dst_ip', '未知')}
            协议: {anomaly.get('protocol_name', '未知')}
            异常类型: {anomaly.get('anomaly_type', '未知')}
            异常分数: {anomaly.get('anomaly_score', 0):.4f}
            置信度: {anomaly.get('confidence', 0):.2%}
            
            请及时检查网络状态。
            """
            msg.attach(MIMEText(body, 'plain'))
            
            # 发送邮件
            self.smtp_server.sendmail(
                self.smtp_config["from_email"],
                self.smtp_config["to_emails"],
                msg.as_string()
            )
            
            self.logger.info(f"已发送邮件告警至 {', '.join(self.smtp_config['to_emails'])}")
            return True
            
        except Exception as e:
            self.logger.error(f"发送邮件告警失败: {str(e)}")
            # 尝试重新初始化连接
            self._init_smtp()
            return False
    
    def _should_send_alert(self, anomaly_type):
        """判断是否应该发送告警（控制频率）"""
        current_time = time.time()
        last_time = self.last_alert_time.get(anomaly_type, 0)
        
        # 检查是否超过告警间隔
        if current_time - last_time >= self.alert_interval:
            self.last_alert_time[anomaly_type] = current_time
            return True
        return False
    
    def log_anomaly(self, anomaly, features=None):
        """
        记录异常检测结果
        
        参数:
            anomaly: 异常信息字典，包含session_id, anomaly_score, anomaly_type等
            features: 导致异常的特征数据
        """
        if not anomaly:
            return False
            
        try:
            # 确保有时间戳
            if "timestamp" not in anomaly:
                anomaly["timestamp"] = time.time()
                
            # 生成异常ID
            anomaly_id = f"{anomaly['session_id']}_{int(anomaly['timestamp'])}"
            anomaly["anomaly_id"] = anomaly_id
            
            # 记录到日志
            log_msg = (
                f"检测到异常 [ID: {anomaly_id}] - "
                f"类型: {anomaly.get('anomaly_type', '未知')}, "
                f"分数: {anomaly.get('anomaly_score', 0):.4f}, "
                f"会话: {anomaly.get('session_id', '未知')}, "
                f"源IP: {anomaly.get('src_ip', '未知')}, "
                f"目的IP: {anomaly.get('dst_ip', '未知')}"
            )
            self.logger.warning(log_msg)
            
            # 保存异常详情到报告文件
            date_str = datetime.fromtimestamp(anomaly["timestamp"]).strftime("%Y%m%d")
            daily_dir = os.path.join(self.reports_dir, date_str)
            os.makedirs(daily_dir, exist_ok=True)
            
            report_path = os.path.join(daily_dir, f"{anomaly_id}.json")
            with open(report_path, "w") as f:
                # 整合异常信息和特征
                report_data = {
                    "anomaly_info": anomaly,
                    "features": features or {}
                }
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"异常详情已保存至 {report_path}")
            
            # 判断是否需要发送告警
            if anomaly.get("anomaly_score", 0) >= self.alert_threshold:
                anomaly_type = anomaly.get("anomaly_type", "unknown")
                if self._should_send_alert(anomaly_type):
                    # 发送邮件告警
                    self._send_email_alert(anomaly)
            
            return True
            
        except Exception as e:
            self.logger.error(f"记录异常时出错: {str(e)}", exc_info=True)
            return False
    
    def get_recent_anomalies(self, hours=24, limit=100):
        """获取最近一段时间的异常记录"""
        recent_anomalies = []
        cutoff_time = time.time() - (hours * 3600)
        
        # 遍历报告目录
        for date_dir in os.listdir(self.reports_dir):
            date_path = os.path.join(self.reports_dir, date_dir)
            if not os.path.isdir(date_path):
                continue
                
            # 遍历当天的异常报告
            for report_file in os.listdir(date_path):
                if not report_file.endswith(".json"):
                    continue
                    
                report_path = os.path.join(date_path, report_file)
                try:
                    with open(report_path, "r") as f:
                        report_data = json.load(f)
                        
                    anomaly_info = report_data.get("anomaly_info", {})
                    if anomaly_info.get("timestamp", 0) >= cutoff_time:
                        recent_anomalies.append(report_data)
                        
                except Exception as e:
                    self.logger.error(f"读取异常报告 {report_path} 时出错: {str(e)}")
        
        # 按时间排序并限制数量
        recent_anomalies.sort(
            key=lambda x: x.get("anomaly_info", {}).get("timestamp", 0), 
            reverse=True
        )
        
        return recent_anomalies[:limit]
    
    def get_status(self):
        """获取组件状态"""
        status = super().get_status()
        status.update({
            "alert_threshold": self.alert_threshold,
            "alert_interval": self.alert_interval,
            "reports_dir": self.reports_dir,
            "smtp_enabled": self.smtp_config["enabled"],
            "recent_anomalies_count": len(self.get_recent_anomalies(hours=1))
        })
        return status
    