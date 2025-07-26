"""
协议特征规范定义，为每种网络协议（TCP/UDP/ICMP等）定义专属特征、模型偏好和异常模式
"""

# 协议号定义（对应IP层protocol字段）
PROTOCOL_NUMBERS = {
    "tcp": 6,
    "udp": 17,
    "icmp": 1,
    "igmp": 2,
    "ipip": 4,
    "tcp6": 6,
    "udp6": 17,
    "icmp6": 58,
    "unknown": -1
}

# 协议特征与模型偏好配置
PROTOCOL_SPECS = {
    # TCP协议 (6)
    6: {
        "name": "tcp",
        "description": "Transmission Control Protocol - 面向连接的可靠传输协议",
        "key_features": [
            "packet_size_mean", "packet_size_std", "packet_size_max",
            "tcp_flags", "tcp_flag_syn", "tcp_flag_ack", "tcp_flag_fin",
            "window_size_mean", "window_size_std",
            "inter_arrival_time_mean", "inter_arrival_time_std",
            "retransmission_count", "segment_size_mean",
            "flow_duration", "packets_per_second",
            "bytes_per_second", "payload_entropy_mean"
        ],
        "model_preference": ["lstm", "xgboost", "random_forest"],  # 优先使用的模型
        "anomaly_patterns": [
            "syn_flood", "port_scan", "unusual_flag_combination",
            "excessive_retransmissions", "unusual_window_size",
            "long_duration", "high_packet_rate"
        ],
        "normal_ranges": {
            "retransmission_count": (0, 5),
            "packets_per_second": (0, 100),
            "window_size_mean": (0, 65535)
        }
    },
    
    # UDP协议 (17)
    17: {
        "name": "udp",
        "description": "User Datagram Protocol - 无连接的不可靠传输协议",
        "key_features": [
            "packet_size_mean", "packet_size_std", "packet_size_max",
            "payload_size_mean", "payload_size_std",
            "flow_duration", "packets_per_second",
            "bytes_per_second", "payload_entropy_mean",
            "port_range", "unique_dest_ports"
        ],
        "model_preference": ["xgboost", "random_forest", "logistic_regression"],
        "anomaly_patterns": [
            "amplification_attack", "dns_amplification",
            "large_payload", "port_flood", "high_packet_rate"
        ],
        "normal_ranges": {
            "packet_size_mean": (0, 1500),
            "packets_per_second": (0, 200),
            "unique_dest_ports": (1, 5)
        }
    },
    
    # ICMP协议 (1)
    1: {
        "name": "icmp",
        "description": "Internet Control Message Protocol - 用于网络控制和诊断",
        "key_features": [
            "packet_size_mean", "packet_size_std",
            "icmp_type", "icmp_code",
            "request_response_ratio", "echo_request_count",
            "echo_reply_count", "flow_duration",
            "packets_per_second", "bytes_per_second"
        ],
        "model_preference": ["random_forest", "xgboost", "logistic_regression"],
        "anomaly_patterns": [
            "icmp_flood", "ping_flood", "unusual_type_code",
            "high_request_response_ratio", "high_packet_rate"
        ],
        "normal_ranges": {
            "request_response_ratio": (0.5, 2),
            "packets_per_second": (0, 10),
            "echo_request_count": (0, 10)
        }
    },
    
    # ICMPv6协议 (58)
    58: {
        "name": "icmp6",
        "description": "ICMP for IPv6 - IPv6网络中的控制和诊断协议",
        "key_features": [
            "packet_size_mean", "packet_size_std",
            "icmp_type", "icmp_code",
            "flow_duration", "packets_per_second",
            "bytes_per_second"
        ],
        "model_preference": ["random_forest", "logistic_regression"],
        "anomaly_patterns": [
            "icmp6_flood", "unusual_type_code", "high_packet_rate"
        ],
        "normal_ranges": {
            "packets_per_second": (0, 10)
        }
    }
}

def get_protocol_spec(protocol_num):
    """
    根据协议号获取协议特征规范
    
    参数:
        protocol_num: 协议号（如6表示TCP）
        
    返回:
        协议特征规范字典，如果协议号未定义则返回默认规范
    """
    if protocol_num in PROTOCOL_SPECS:
        return PROTOCOL_SPECS[protocol_num].copy()
    
    # 未知协议的默认规范
    return {
        "name": "unknown",
        "description": "Unknown protocol - 未识别的网络协议",
        "key_features": [
            "packet_size_mean", "packet_size_std",
            "flow_duration", "packets_per_second",
            "bytes_per_second"
        ],
        "model_preference": ["xgboost", "random_forest"],
        "anomaly_patterns": ["unusual_size", "high_packet_rate"],
        "normal_ranges": {
            "packet_size_mean": (0, 1500),
            "packets_per_second": (0, 100)
        }
    }

def get_protocol_number(protocol_name):
    """
    根据协议名称获取协议号
    
    参数:
        protocol_name: 协议名称（如"tcp"）
        
    返回:
        协议号，如果名称未定义则返回-1
    """
    return PROTOCOL_NUMBERS.get(protocol_name.lower(), -1)

def get_all_protocols():
    """获取所有支持的协议列表"""
    return PROTOCOL_SPECS

def get_protocol_key_features(protocol_num):
    """获取指定协议的关键特征列表"""
    spec = get_protocol_spec(protocol_num)
    return spec["key_features"]

def get_protocol_model_preference(protocol_num):
    """获取指定协议的模型偏好列表"""
    spec = get_protocol_spec(protocol_num)
    return spec["model_preference"]

def is_feature_relevant(protocol_num, feature_name):
    """判断特征是否与指定协议相关"""
    key_features = get_protocol_key_features(protocol_num)
    return feature_name in key_features