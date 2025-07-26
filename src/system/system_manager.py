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
    
    _instance = None  # 类属性，用于存储单例实例
    _stopping = False  # 类级别属性，表示系统是否正在停止
    
    def __init__(self, config=None):
        super().__init__()
        SystemManager._instance = self  # 保存实例引用
        
        # 使用单例模式获取ConfigManager实例
        self.config = config or ConfigManager()
        
        # 初始化日志记录器
        self.logger = get_logger("system.manager")
        
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
    
    def _initialize_components(self) -> None:
        """初始化所有系统组件"""
        try:
            # 监控组件（独立运行）
            self._components["monitor"] = SystemMonitor()
            
            # 数据采集组件
            self._components["packet_capture"] = PacketCapture()
            
            # 会话跟踪组件
            self._components["session_tracker"] = SessionTracker()
            
            # 流量分析组件
            self._components["traffic_analyzer"] = TrafficAnalyzer()
            
            # 检测与告警组件
            self._components["anomaly_detector"] = AnomalyDetector(
                config=self.config
            )
            
            # 模型相关组件
            self._components["model_factory"] = ModelFactory(
                config=self.config
            )
            self._components["model_selector"] = ModelSelector(
                config=self.config
            )
            
            self._components["alert_manager"] = AlertManager(
                config=self.config
            )
            
            # 反馈处理组件
            self._components["feedback_processor"] = FeedbackProcessor(
                config=self.config
            )
            
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
            
        component = self._components[component_name]
        # 检查组件是否是BaseComponent的实例，如果不是则跳过
        if not isinstance(component, BaseComponent):
            self.logger.warning(f"组件 {component_name} 不是BaseComponent的实例")
            return True
            
        if component.is_running:
            self.logger.debug(f"组件 {component_name} 已在运行中")
            return True
            
        try:
            # 特殊组件可能需要额外参数
            if component_name == "packet_capture":
                # 设置网络接口和过滤规则
                interface = self.config.get("network.interface")
                filter_rule = self.config.get("network.filter")
                if interface:
                    component.set_interface(interface)
                if filter_rule:
                    component.set_filter(filter_rule)
                component.start()
            elif component_name == "traffic_analyzer":
                component.start(session_tracker=self._components["session_tracker"])
            elif component_name == "anomaly_detector":
                component.start()
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
        self.logger.debug(f"正在停止组件 {component_name} (类型: {type(component).__name__})")
        
        # 检查组件是否是BaseComponent的实例，如果不是则跳过
        if not isinstance(component, BaseComponent):
            self.logger.debug(f"组件 {component_name} 不是BaseComponent的实例，跳过停止")
            return True
            
        try:
            # 检查组件是否正在运行
            if not component.is_running:
                self.logger.debug(f"组件 {component_name} 未运行，无需停止")
                return True
                
            self.logger.debug(f"调用组件 {component_name} 的stop方法")
            component.stop()
            self.logger.info(f"组件 {component_name} 已停止")
            return True
        except Exception as e:
            self.logger.error(f"停止组件 {component_name} 失败: {str(e)}", exc_info=True)
            try:
                component.record_error(e)
            except Exception as inner_e:
                self.logger.error(f"记录组件 {component_name} 错误信息失败: {str(inner_e)}", exc_info=True)
            return False
    
    def _monitor_components(self) -> None:
        """监控组件运行状态"""
        while self.is_running:
            try:
                # 检查所有组件状态
                for name, component in self._components.items():
                    # 跳过不需要监控的组件
                    if name in ["config_manager"]:
                        continue
                        
                    # 检查组件是否运行正常
                    if component.is_running:
                        continue
                        
                    # 如果系统正在停止过程中，不尝试重启组件
                    if self._stopping:
                        continue
                        
                    status = component.get_status()
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
        
        # 按依赖顺序重启组件
        start_order = self._get_component_start_order()
        if component_name in start_order:
            # 记录重启时间
            self._restart_history[component_name].append(now)
            
            # 重启组件
            if self._start_component(component_name):
                self.logger.info(f"组件 {component_name} 重启成功")
                return True
            else:
                self.logger.error(f"组件 {component_name} 重启失败")
                return False
        else:
            self.logger.warning(f"未知组件 {component_name}，无法重启")
            return False
    
    def start(self, interface: Optional[str] = None, bpf_filter: Optional[str] = None) -> bool:
        """
        启动系统管理器和所有组件
        
        Args:
            interface: 网络接口
            bpf_filter: BPF过滤规则
            
        Returns:
            是否启动成功
        """
        if self.is_running:
            self.logger.warning("系统已在运行中")
            return False
            
        try:
            # 设置网络接口和过滤规则（如果提供）
            if interface:
                self.config.set("network.interface", interface)
            if bpf_filter:
                self.config.set("network.bpf_filter", bpf_filter)
                
            # 确定组件启动顺序
            start_order = self._get_component_start_order()
            self.logger.debug(f"组件启动顺序: {start_order}")
            
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
            
        # 设置停止标志
        self._stopping = True
            
        try:
            # 按与启动相反的顺序停止组件
            start_order = self._get_component_start_order()
            stop_order = list(reversed(start_order))
            self.logger.debug(f"组件停止顺序: {stop_order}")
            
            for component_name in stop_order:
                try:
                    self.logger.debug(f"正在停止组件: {component_name}")
                    self._stop_component(component_name)
                    self.logger.debug(f"组件 {component_name} 停止完成")
                except Exception as e:
                    self.logger.error(f"停止组件 {component_name} 时出现异常: {str(e)}", exc_info=True)
            
            # 等待监控线程结束
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self.logger.debug("正在等待监控线程结束")
                self._monitoring_thread.join(timeout=5)
                if self._monitoring_thread.is_alive():
                    self.logger.warning("监控线程未能正常终止")
            
            # 最后停止系统管理器自身
            super().stop()
            self.logger.debug("系统管理器自身已停止")
            
            self.logger.info("系统已完全停止")
            
        except Exception as e:
            self.logger.error(f"系统停止过程中出现异常: {str(e)}", exc_info=True)
        
        finally:
            self._stopping = False  # 确保在最后将_stopping标志重置为False
            # 可以在这里添加其他清理操作，如释放资源、关闭连接等
            # 例如：self.cleanup_resources()
            
        # 可以继续执行其他finally块的内容

    def get_component(self, component_name: str) -> Optional[BaseComponent]:
        """获取指定组件实例"""
        return self._components.get(component_name)

    def get_all_components(self) -> Dict[str, BaseComponent]:
        """获取所有组件"""
        return self._components.copy()

    def get_system_status(self) -> Dict[str, any]:
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