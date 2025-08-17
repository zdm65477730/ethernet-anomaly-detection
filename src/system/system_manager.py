"""
系统管理器
"""
import threading
import time
import os
import traceback
from typing import Dict, Any, Optional, List
from src.utils.logger import get_logger
from src.config.config_manager import ConfigManager
from src.capture.packet_capture import PacketCapture
from src.capture.session_tracker import SessionTracker
from src.analysis.traffic_analyzer import TrafficAnalyzer
from src.detection.anomaly_detector import AnomalyDetector
from src.models.model_factory import ModelFactory
from src.models.model_selector import ModelSelector
from src.detection.alert_manager import AlertManager
from src.detection.feedback_processor import FeedbackProcessor
from src.monitoring.monitor import SystemMonitor
from src.system.base_component import BaseComponent

logger = get_logger(__name__)

class SystemManager(BaseComponent):
    """系统管理器，负责初始化、启动、停止所有组件，并监控组件状态"""
    
    _instance = None  # 类属性，用于存储单例实例
    _stopping = False  # 类级别属性，表示系统是否正在停止
    
    def __init__(self, config=None):
        super().__init__()
        SystemManager._instance = self  # 保存实例引用
        
        # 使用单例模式获取ConfigManager实例
        self.config = config or ConfigManager()
        
        # 初始化日志记录器
        self.logger = get_logger("system.manager")
        
        # 清除任何可能存在的停止标志文件
        stop_flag_file = ".system_stop_flag"
        try:
            if os.path.exists(stop_flag_file):
                os.remove(stop_flag_file)
        except Exception as e:
            self.logger.warning(f"清除停止标志文件失败: {e}")
        
        # 重置SystemMonitor的强制停止标志
        try:
            from src.monitoring.monitor import SystemMonitor
            SystemMonitor._force_stopped = False
            SystemMonitor._global_stopped = False
        except ImportError:
            self.logger.warning("无法导入SystemMonitor类")
        
        # 组件字典
        self._components: Dict[str, BaseComponent] = {}
        
        # 组件依赖关系
        self._component_dependencies: Dict[str, List[str]] = {
            "session_tracker": ["packet_capture"],
            "traffic_analyzer": ["session_tracker"],
            "anomaly_detector": ["traffic_analyzer"],
            "alert_manager": ["anomaly_detector"],
            "feedback_processor": ["alert_manager"],
            "monitor": []
        }
        
        # 组件重启次数限制
        self._max_restarts: Dict[str, int] = {
            "packet_capture": 3,
            "session_tracker": 3,
            "traffic_analyzer": 3,
            "anomaly_detector": 3,
            "alert_manager": 3,
            "feedback_processor": 3,
            "monitor": 3
        }
        
        # 系统停止标志
        self._stopping = False
        
        # 组件重启历史
        self._restart_history: Dict[str, List[float]] = {}  # 组件重启历史
        
        # 故障重启配置
        self._max_restarts_count = 5  # 最大重启次数
        self._restart_window = 60  # 重启时间窗口(秒)
        
        # 组件监控间隔（秒）
        self._monitor_interval = 1
        
        # 监控线程
        self._monitoring_thread: Optional[threading.Thread] = None
        
        # 初始化所有组件
        self._initialize_components()
    
    def _start_monitoring_thread(self):
        """启动组件监控线程"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self.logger.debug("组件监控线程已在运行中")
            return
            
        self._monitoring_thread = threading.Thread(
            target=self._monitor_components,
            daemon=True
        )
        self._monitoring_thread.start()
        self.logger.debug("组件监控线程已启动")
    
    def _stop_monitoring_thread(self):
        """停止组件监控线程"""
        if not self._monitoring_thread or not self._monitoring_thread.is_alive():
            self.logger.debug("组件监控线程未运行")
            return
            
        # 检查是否在监控线程内部调用
        if threading.current_thread() is self._monitoring_thread:
            self.logger.debug("在监控线程内部调用停止，无需等待自身")
            return
            
        self.logger.debug("正在停止组件监控线程")
        # 设置停止标志
        self._stopping = True
        SystemManager._stopping = True
        
        # 等待线程结束
        self._monitoring_thread.join(timeout=5)
        if self._monitoring_thread.is_alive():
            self.logger.warning("组件监控线程未能在超时时间内终止")
        else:
            self.logger.debug("组件监控线程已正常终止")
    
    def _initialize_components(self):
        """初始化所有系统组件"""
        try:
            self.logger.info("开始初始化系统组件")
            
            # 初始化监控组件（如果启用）
            if self.config.get("monitoring.enabled", True):
                self._components["monitor"] = SystemMonitor(self.config)
            
            # 按依赖顺序初始化组件
            # 1. 初始化数据捕获组件
            self._components["packet_capture"] = PacketCapture(self.config)
            
            # 2. 初始化会话跟踪组件
            self._components["session_tracker"] = SessionTracker(self.config)
            
            # 3. 初始化异常检测组件
            self._components["anomaly_detector"] = AnomalyDetector(self.config)
            
            # 4. 初始化流量分析组件
            session_tracker = self._components["session_tracker"]
            anomaly_detector = self._components["anomaly_detector"]
            self._components["traffic_analyzer"] = TrafficAnalyzer(
                session_tracker=session_tracker,
                anomaly_detector=anomaly_detector
            )
            
            # 5. 初始化模型相关组件
            self._components["model_factory"] = ModelFactory(self.config)
            self._components["model_selector"] = ModelSelector(self.config)
            
            # 6. 初始化告警和反馈组件
            self._components["alert_manager"] = AlertManager(self.config)
            self._components["feedback_processor"] = FeedbackProcessor(self.config)
            
            self.logger.info("所有组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"组件初始化失败: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise
    
    def _get_component_start_order(self) -> List[str]:
        """获取组件启动顺序（考虑依赖关系的拓扑排序）"""
        # 构建依赖图并进行拓扑排序
        visited = set()
        order = []
        
        def dfs(component_name):
            if component_name in visited:
                return
            visited.add(component_name)
            
            # 先启动依赖组件
            deps = self._component_dependencies.get(component_name, [])
            for dep in deps:
                if dep in self._components:
                    dfs(dep)
            
            # 再启动当前组件
            order.append(component_name)
        
        # 对所有组件执行DFS
        for component in self._components:
            dfs(component)
            
        return order

    def _start_component(self, component_name: str) -> bool:
        """启动单个组件"""
        if component_name not in self._components:
            self.logger.error(f"组件 {component_name} 不存在")
            return False
            
        try:
            component = self._components[component_name]
            if component.is_running:
                self.logger.warning(f"组件 {component_name} 已在运行中")
                return True
                
            self.logger.debug(f"正在启动组件: {component_name}")
            
            # 启动组件
            result = component.start()
            
            # 检查启动结果
            if result is False:
                self.logger.error(f"组件 {component_name} 启动失败")
                return False
                
            self.logger.info(f"组件 {component_name} 启动成功")
            return True
                
        except Exception as e:
            self.logger.error(f"启动组件 {component_name} 时出现异常: {str(e)}", exc_info=True)
            return False

    def _stop_component(self, component_name: str) -> bool:
        """停止单个组件"""
        if component_name not in self._components:
            self.logger.error(f"组件 {component_name} 不存在")
            return False
            
        try:
            component = self._components[component_name]
            if not component.is_running:
                self.logger.debug(f"组件 {component_name} 未运行")
                return True
                
            self.logger.debug(f"正在停止组件: {component_name}")
            component.stop()
            self.logger.info(f"组件 {component_name} 已停止")
            return True
            
        except Exception as e:
            self.logger.error(f"停止组件 {component_name} 时出现异常: {str(e)}", exc_info=True)
            return False

    def start_system(self, **kwargs):
        """启动系统及所有组件"""
        try:
            self.logger.info("开始启动系统")
            
            # 启动监控组件（如果启用）
            if "monitor" in self._components:
                if self._start_component("monitor"):
                    self.logger.info("组件 monitor 启动成功")
                else:
                    self.logger.error("组件 monitor 启动失败")
            
            # 按顺序启动其他组件
            component_order = [
                "packet_capture", 
                "session_tracker", 
                "traffic_analyzer", 
                "anomaly_detector", 
                "model_factory", 
                "model_selector",
                "alert_manager", 
                "feedback_processor"
            ]
            
            self.logger.info(f"组件启动顺序: {component_order}")
            
            for component_name in component_order:
                if component_name in self._components:
                    # 启动组件
                    if self._start_component(component_name):
                        self.logger.info(f"组件 {component_name} 启动成功")
                    else:
                        self.logger.error(f"组件 {component_name} 启动失败")
                        # 可以选择是否继续启动其他组件
                        continue
                else:
                    self.logger.debug(f"组件 {component_name} 不存在，跳过启动")
            
            # 连接组件
            self._connect_components()
            
            self.logger.info("系统启动完成")
            self._is_running = True
            return True
            
        except Exception as e:
            self.logger.error(f"启动系统时出现异常: {str(e)}", exc_info=True)
            return False
    
    def _connect_components(self):
        """连接各个组件"""
        try:
            self.logger.info("开始建立组件连接")
            
            # 连接packet_capture和session_tracker
            packet_capture = self._components.get("packet_capture")
            session_tracker = self._components.get("session_tracker")
            if packet_capture and session_tracker:
                packet_capture.session_tracker = session_tracker
                self.logger.debug("已连接 packet_capture 到 session_tracker")
            
            # 连接session_tracker和traffic_analyzer
            traffic_analyzer = self._components.get("traffic_analyzer")
            if session_tracker and traffic_analyzer:
                # 检查方法是否存在
                if hasattr(traffic_analyzer, 'enqueue_session'):
                    session_tracker.set_data_callback(traffic_analyzer.enqueue_session)
                    self.logger.debug("已连接 session_tracker 到 traffic_analyzer")
                else:
                    self.logger.warning("无法建立session_tracker到traffic_analyzer的连接：traffic_analyzer或enqueue_session方法不存在")
            else:
                self.logger.warning("无法建立session_tracker到traffic_analyzer的连接：组件不存在")
            
            # 连接traffic_analyzer和anomaly_detector
            anomaly_detector = self._components.get("anomaly_detector")
            if traffic_analyzer and anomaly_detector:
                # 直接共享特征队列
                if hasattr(traffic_analyzer, 'features_queue') and hasattr(anomaly_detector, 'features_queue'):
                    traffic_analyzer.features_queue = anomaly_detector.features_queue
                    self.logger.debug("已连接 traffic_analyzer 到 anomaly_detector")
                else:
                    self.logger.warning("无法建立traffic_analyzer到anomaly_detector的连接：组件缺少features_queue属性")
            else:
                self.logger.warning("无法建立traffic_analyzer到anomaly_detector的连接：组件不存在")
                
        except Exception as e:
            self.logger.error(f"连接组件时出错: {e}")

    def _monitor_components(self):
        """监控组件运行状态"""
        try:
            self.logger.debug("组件监控线程开始运行")
            stop_flag_file = ".system_stop_flag"
            
            while not self._stopping and not SystemManager._stopping:
                # 检查停止标志文件
                if os.path.exists(stop_flag_file):
                    self.logger.info("检测到系统停止标志文件，正在停止系统")
                    self.stop()
                    return
                
                try:
                    # 检查是否是离线文件处理模式且处理已完成
                    packet_capture = self._components.get("packet_capture")
                    if (packet_capture and packet_capture.mode == "offline" and 
                        hasattr(packet_capture, 'offline_processing_complete') and 
                        packet_capture.offline_processing_complete):
                        self.logger.info("离线文件处理完成，正在停止系统")
                        self.stop()
                        return
                    
                    # 如果系统正在停止过程中，不尝试重启组件
                    if self._stopping or SystemManager._stopping:
                        self.logger.debug("系统正在停止过程中，监控线程即将退出")
                        return  # 直接返回而不是break
                    
                    # 检查所有组件状态
                    for name, component in self._components.items():
                        # 检查停止标志文件
                        if os.path.exists(stop_flag_file):
                            self.logger.info("检测到系统停止标志文件，正在停止系统")
                            self.stop()
                            return
                            
                        # 跳过不需要监控的组件
                        if name == "config_manager":
                            continue
                            
                        # 检查组件是否运行正常
                        if component.is_running:
                            continue
                            
                        # 如果系统正在停止过程中，不尝试重启任何组件
                        if self._stopping or SystemManager._stopping:
                            self.logger.debug("系统正在停止过程中，跳过组件重启")
                            return  # 直接返回而不是break
                            
                        status = component.get_status()
                        self.logger.warning(f"组件 {name} 未运行，状态: {status}")
                        
                        # 尝试重启组件
                        self.logger.debug(f"尝试重启组件: {name}")
                        success = self._attempt_restart(name)
                        if success:
                            self.logger.info(f"组件 {name} 重启成功")
                        else:
                            self.logger.error(f"组件 {name} 重启失败")
                    
                    # 更频繁地检查停止信号
                    for _ in range(int(self._monitor_interval * 10)):  # 分成更小的时间段检查
                        # 检查停止标志文件
                        if os.path.exists(stop_flag_file):
                            self.logger.info("检测到系统停止标志文件，正在停止系统")
                            self.stop()
                            return
                            
                        # 检查是否是离线文件处理模式且处理已完成
                        packet_capture = self._components.get("packet_capture")
                        if (packet_capture and packet_capture.mode == "offline" and 
                            hasattr(packet_capture, 'offline_processing_complete') and 
                            packet_capture.offline_processing_complete):
                            self.logger.info("离线文件处理完成，正在停止系统")
                            self.stop()
                            return
                            
                        # 检查系统是否正在停止
                        if self._stopping or SystemManager._stopping:
                            self.logger.debug("系统正在停止过程中，监控线程即将退出")
                            return
                        time.sleep(0.1)  # 每0.1秒检查一次
                        
                except Exception as e:
                    self.logger.error(f"组件监控循环出错: {str(e)}", exc_info=True)
                    # 即使出错也要检查停止标志
                    if os.path.exists(stop_flag_file):
                        self.logger.info("检测到系统停止标志文件，正在停止系统")
                        self.stop()
                        return
                        
                    # 检查是否是离线文件处理模式且处理已完成
                    packet_capture = self._components.get("packet_capture")
                    if (packet_capture and packet_capture.mode == "offline" and 
                        hasattr(packet_capture, 'offline_processing_complete') and 
                        packet_capture.offline_processing_complete):
                        self.logger.info("离线文件处理完成，正在停止系统")
                        self.stop()
                        return
                        
                    # 即使出错也要检查停止标志
                    if self._stopping or SystemManager._stopping:
                        self.logger.debug("系统正在停止过程中，监控线程即将退出")
                        return
                    time.sleep(1)
            
            self.logger.debug("组件监控线程已退出")
            
        except Exception as e:
            self.logger.error(f"组件监控线程发生未捕获异常: {str(e)}", exc_info=True)
        finally:
            self.logger.debug("组件监控线程结束")
    
    def _attempt_restart(self, component_name: str) -> bool:
        """尝试重启组件，并检查重启次数限制"""
        # 清理过期的重启记录
        now = time.time()
        if component_name not in self._restart_history:
            self._restart_history[component_name] = []
        
        # 确保_restart_history[component_name]是列表而不是字典
        if not isinstance(self._restart_history[component_name], list):
            self._restart_history[component_name] = []
        
        self._restart_history[component_name] = [
            t for t in self._restart_history[component_name]
            if now - t < self._restart_window
        ]
        
        # 检查重启次数是否超过限制
        max_restarts = self._max_restarts.get(component_name, 3)  # 默认3次
        if len(self._restart_history[component_name]) >= max_restarts:
            self.logger.error(
                f"组件 {component_name} 在 {self._restart_window} 秒内已重启 "
                f"{max_restarts} 次，达到上限，停止尝试"
            )
            return False
            
        # 记录本次重启时间
        self._restart_history[component_name].append(now)
        
        # 重启组件
        try:
            self.logger.info(f"正在重启组件 {component_name}")
            component = self._components[component_name]
            
            # 先停止组件
            if component.is_running:
                component.stop()
                
            # 再启动组件
            success = False
            if component_name == "packet_capture":
                # PacketCapture组件需要特殊处理
                success = component.start()
            else:
                success = component.start()
                
            return success
        except Exception as e:
            self.logger.error(f"重启组件 {component_name} 失败: {str(e)}", exc_info=True)
            return False

    def stop(self):
        """停止系统"""
        if not self._is_running:
            self.logger.info("系统未运行")
            return True
            
        self.logger.info("开始停止系统")
        self._is_running = False
        
        # 按相反顺序停止组件
        start_order = self._get_component_start_order()
        stop_order = list(reversed(start_order))  # 反转启动顺序作为停止顺序
        self.logger.debug(f"组件停止顺序: {stop_order}")
        
        for component_name in stop_order:
            component = self._components.get(component_name)
            if component:
                try:
                    # 特殊处理session_tracker，确保会话被刷新
                    if component_name == "session_tracker":
                        self.logger.info(f"正在刷新 {component_name} 的会话")
                        component._flush_all_sessions()
                    
                    self._stop_component(component, component_name)
                except Exception as e:
                    self.logger.error(f"停止组件 {component_name} 时出现异常: {e}")
                    self.logger.debug(traceback.format_exc())
        
        # 清理资源
        self._components.clear()
        
        # 清除停止标志文件
        try:
            if os.path.exists(self.STOP_FLAG_FILE):
                os.remove(self.STOP_FLAG_FILE)
        except Exception as e:
            self.logger.warning(f"清除停止标志文件失败: {e}")
        
        self.logger.info("系统已完全停止")
        return True

    def get_component(self, component_name: str) -> Optional[BaseComponent]:
        """获取指定组件实例"""
        return self._components.get(component_name)

    def get_all_components(self) -> Dict[str, BaseComponent]:
        """获取所有组件"""
        return self._components.copy()

    def get_system_status(self) -> Dict[str, Any]:
        """获取整个系统的状态信息"""
        system_status = super().get_status()
        
        # 添加各组件状态
        components_status = {}
        for name, component in self._components.items():
            components_status[name] = component.get_status()
        
        system_status.update({
            "component_count": len(self._components),
            "running_components": sum(1 for c in self._components.values() if c.is_running),
            "components": components_status,
            "restart_history": self._restart_history,
            "monitor_interval": self._monitor_interval
        })
        
        return system_status

    def restart_component(self, component_name: str) -> bool:
        """手动重启指定组件"""
        self.logger.info(f"收到手动重启组件 {component_name} 的请求")
        
        # 先停止组件
        self._stop_component(component_name)
        # 清理重启历史，允许立即重启
        if component_name in self._restart_history:
            del self._restart_history[component_name]
        # 尝试重启
        return self._attempt_restart(component_name)
