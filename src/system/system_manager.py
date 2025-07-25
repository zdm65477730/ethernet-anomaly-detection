import time
import threading
import traceback
from typing import Dict, List, Optional, Tuple
from src.system.base_component import BaseComponent
from src.utils.logger import get_logger
from src.config.config_manager import ConfigManager
from src.monitoring.monitor import SystemMonitor
from src.capture.packet_capture import PacketCapture
from src.capture.session_tracker import SessionTracker
from src.capture.traffic_analyzer import TrafficAnalyzer
from src.detection.anomaly_detector import AnomalyDetector
from src.detection.alert_manager import AlertManager
from src.detection.feedback_processor import FeedbackProcessor
from src.models.model_factory import ModelFactory
from src.models.model_selector import ModelSelector

class SystemManager(BaseComponent):
    """系统管理器，负责初始化、启动、停止所有组件，并监控组件状态"""
    
    def __init__(self, config=None):
        super().__init__()
        self.logger = get_logger("system.manager")
        self.config = config or ConfigManager()
        
        # 组件字典
        self._components: Dict[str, BaseComponent] = {}
        
        # 组件依赖关系
        self._component_dependencies: Dict[str, List[str]] = {
            "session_tracker": ["packet_capture"],
            "traffic_analyzer": ["session_tracker"],
            "anomaly_detector": ["traffic_analyzer", "model_factory", "model_selector"],
            "alert_manager": ["anomaly_detector"],
            "feedback_processor": ["anomaly_detector"]
        }
        
        # 监控线程
        self._monitoring_thread = None
        self._monitor_interval = 5  # 组件监控间隔(秒)
        
        # 故障重启配置
        self._max_restarts = 5  # 最大重启次数
        self._restart_window = 60  # 重启时间窗口(秒)
        self._restart_history: Dict[str, List[float]] = {}  # 组件重启历史
        
        # 初始化所有组件
        self._initialize_components()

    def _initialize_components(self) -> None:
        """初始化所有系统组件"""
        try:
            # 基础组件
            self._components["config_manager"] = self.config
            self._components["system_monitor"] = SystemMonitor(config=self.config)
            self._components["model_factory"] = ModelFactory(config=self.config)
            self._components["model_selector"] = ModelSelector(config=self.config)
            
            # 流量捕获与会话跟踪组件
            self._components["packet_capture"] = PacketCapture(
                interface=self.config.get("network.interface"),
                filter=self.config.get("network.filter"),
                config=self.config
            )
            self._components["session_tracker"] = SessionTracker(config=self.config)
            
            # 流量分析组件
            self._components["traffic_analyzer"] = TrafficAnalyzer(config=self.config)
            
            # 检测与告警组件
            self._components["anomaly_detector"] = AnomalyDetector(
                model_factory=self._components["model_factory"],
                model_selector=self._components["model_selector"],
                config=self.config
            )
            self._components["alert_manager"] = AlertManager(config=self.config)
            self._components["feedback_processor"] = FeedbackProcessor(
                feature_extractor=self._components["traffic_analyzer"].stat_extractor,
                config=self.config
            )
            
            self.logger.info(f"已初始化 {len(self._components)} 个组件")
            
        except Exception as e:
            self.logger.error(f"组件初始化失败: {str(e)}", exc_info=True)
            raise

    def _get_component_start_order(self) -> List[str]:
        """
        确定组件启动顺序，确保依赖组件先启动
        
        返回:
            组件名称列表，按启动顺序排列
        """
        visited = set()
        order = []
        
        def dfs(component):
            if component not in visited:
                visited.add(component)
                # 先启动依赖组件
                for dep in self._component_dependencies.get(component, []):
                    dfs(dep)
                order.append(component)
        
        # 对所有组件执行深度优先搜索，确定启动顺序
        for component in self._components:
            dfs(component)
            
        return order

    def _start_component(self, component_name: str) -> bool:
        """启动单个组件"""
        if component_name not in self._components:
            self.logger.error(f"组件 {component_name} 不存在")
            return False
            
        component = self._components[component_name]
        if component.is_running:
            self.logger.debug(f"组件 {component_name} 已在运行中")
            return True
            
        try:
            # 特殊组件可能需要额外参数
            if component_name == "traffic_analyzer":
                component.start(session_tracker=self._components["session_tracker"])
            elif component_name == "anomaly_detector":
                component.start(
                    feature_queue=self._components["traffic_analyzer"]._feature_queue,
                    alert_manager=self._components["alert_manager"],
                    feedback_processor=self._components["feedback_processor"]
                )
            else:
                component.start()
                
            self.logger.info(f"组件 {component_name} 已启动")
            return True
            
        except Exception as e:
            self.logger.error(f"启动组件 {component_name} 失败: {str(e)}", exc_info=True)
            component.record_error(e)
            return False

    def _stop_component(self, component_name: str) -> bool:
        """停止单个组件"""
        if component_name not in self._components:
            self.logger.error(f"组件 {component_name} 不存在")
            return False
            
        component = self._components[component_name]
        if not component.is_running:
            self.logger.debug(f"组件 {component_name} 已停止")
            return True
            
        try:
            component.stop()
            self.logger.info(f"组件 {component_name} 已停止")
            return True
            
        except Exception as e:
            self.logger.error(f"停止组件 {component_name} 失败: {str(e)}", exc_info=True)
            component.record_error(e)
            return False

    def _monitor_components(self) -> None:
        """监控组件状态，自动重启故障组件"""
        while self.is_running:
            try:
                # 检查所有组件状态
                for name, component in self._components.items():
                    status = component.get_status()
                    
                    # 检查组件是否运行正常
                    if not status["is_running"] and name not in ["config_manager"]:
                        self.logger.warning(f"组件 {name} 未运行，状态: {status}")
                        
                        # 尝试重启组件
                        self._attempt_restart(name)
                
                time.sleep(self._monitor_interval)
                
            except Exception as e:
                self.logger.error(f"组件监控循环出错: {str(e)}", exc_info=True)
                time.sleep(1)

    def _attempt_restart(self, component_name: str) -> bool:
        """尝试重启组件，并检查重启次数限制"""
        # 清理过期的重启记录
        now = time.time()
        if component_name not in self._restart_history:
            self._restart_history[component_name] = []
        
        self._restart_history[component_name] = [
            t for t in self._restart_history[component_name]
            if now - t < self._restart_window
        ]
        
        # 检查重启次数是否超过限制
        if len(self._restart_history[component_name]) >= self._max_restarts:
            self.logger.error(
                f"组件 {component_name} 在 {self._restart_window} 秒内已重启 "
                f"{self._max_restarts} 次，达到上限，停止尝试"
            )
            return False
        
        # 按依赖顺序重启组件
        start_order = self._get_component_start_order()
        component_index = start_order.index(component_name) if component_name in start_order else -1
        
        if component_index == -1:
            # 直接重启单个组件
            success = self._start_component(component_name)
        else:
            # 按顺序重启从依赖组件到目标组件
            success = True
            for name in start_order[:component_index + 1]:
                if not self._start_component(name):
                    success = False
                    break
        
        if success:
            # 记录重启时间
            self._restart_history[component_name].append(now)
            self.logger.info(f"组件 {component_name} 重启成功")
        else:
            self.logger.error(f"组件 {component_name} 重启失败")
            
        return success

    def start(self) -> bool:
        """启动所有组件"""
        if self.is_running:
            self.logger.warning("系统已在运行中")
            return True
            
        try:
            # 确定组件启动顺序
            start_order = self._get_component_start_order()
            self.logger.info(f"组件启动顺序: {start_order}")
            
            # 按顺序启动所有组件
            all_success = True
            for component_name in start_order:
                if not self._start_component(component_name):
                    all_success = False
                    # 关键组件启动失败，停止启动后续组件
                    if component_name in ["packet_capture", "session_tracker", "anomaly_detector"]:
                        self.logger.error(f"关键组件 {component_name} 启动失败，停止系统启动")
                        self.stop()
                        return False
            
            # 启动系统管理器自身
            super().start()
            
            # 启动组件监控线程
            self._monitoring_thread = threading.Thread(
                target=self._monitor_components,
                daemon=True
            )
            self._monitoring_thread.start()
            
            self.logger.info("系统启动完成" + ("，部分组件启动失败" if not all_success else ""))
            return all_success
            
        except Exception as e:
            self.logger.error(f"系统启动失败: {str(e)}", exc_info=True)
            self.stop()
            return False

    def stop(self) -> None:
        """停止所有组件（按启动相反顺序）"""
        if not self.is_running:
            self.logger.warning("系统已停止")
            return
            
        # 先停止监控线程
        super().stop()
        
        # 按与启动相反的顺序停止组件
        start_order = self._get_component_start_order()
        stop_order = reversed(start_order)
        
        for component_name in stop_order:
            self._stop_component(component_name)
        
        # 等待监控线程结束
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5)
        
        self.logger.info("系统已完全停止")

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
