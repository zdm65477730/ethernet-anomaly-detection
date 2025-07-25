"""异常网络流量样本数据"""

# 端口扫描攻击数据包（异常特征：短时间内连接多个端口）
port_scan_packets = [
    {
        "timestamp": 1620000100.0,
        "src_ip": "10.0.0.10",
        "dst_ip": "192.168.1.100",
        "src_port": 49876,
        "dst_port": 21,   # FTP
        "protocol": 6,    # TCP
        "length": 60,
        "tcp_flags": {"SYN": True},
        "payload": ""
    },
    {
        "timestamp": 1620000100.1,
        "src_ip": "10.0.0.10",
        "dst_ip": "192.168.1.100",
        "src_port": 49876,
        "dst_port": 22,   # SSH
        "protocol": 6,    # TCP
        "length": 60,
        "tcp_flags": {"SYN": True},
        "payload": ""
    },
    {
        "timestamp": 1620000100.2,
        "src_ip": "10.0.0.10",
        "dst_ip": "192.168.1.100",
        "src_port": 49876,
        "dst_port": 23,   # Telnet
        "protocol": 6,    # TCP
        "length": 60,
        "tcp_flags": {"SYN": True},
        "payload": ""
    },
    {
        "timestamp": 1620000100.3,
        "src_ip": "10.0.0.10",
        "dst_ip": "192.168.1.100",
        "src_port": 49876,
        "dst_port": 80,   # HTTP
        "protocol": 6,    # TCP
        "length": 60,
        "tcp_flags": {"SYN": True},
        "payload": ""
    },
    {
        "timestamp": 1620000100.4,
        "src_ip": "10.0.0.10",
        "dst_ip": "192.168.1.100",
        "src_port": 49876,
        "dst_port": 443,  # HTTPS
        "protocol": 6,    # TCP
        "length": 60,
        "tcp_flags": {"SYN": True},
        "payload": ""
    },
    {
        "timestamp": 1620000100.5,
        "src_ip": "10.0.0.10",
        "dst_ip": "192.168.1.100",
        "src_port": 49876,
        "dst_port": 3389, # RDP
        "protocol": 6,    # TCP
        "length": 60,
        "tcp_flags": {"SYN": True},
        "payload": ""
    }
]

# DDoS攻击数据包（异常特征：大量小数据包）
ddos_attack_packets = [
    {
        "timestamp": 1620000200.0,
        "src_ip": "198.51.100.20",
        "dst_ip": "192.168.1.100",
        "src_port": 51234,
        "dst_port": 80,
        "protocol": 6,    # TCP
        "length": 40,
        "tcp_flags": {"SYN": True},
        "payload": ""
    },
    {
        "timestamp": 1620000200.001,
        "src_ip": "198.51.100.21",
        "dst_ip": "192.168.1.100",
        "src_port": 51235,
        "dst_port": 80,
        "protocol": 6,    # TCP
        "length": 40,
        "tcp_flags": {"SYN": True},
        "payload": ""
    },
    {
        "timestamp": 1620000200.002,
        "src_ip": "198.51.100.22",
        "dst_ip": "192.168.1.100",
        "src_port": 51236,
        "dst_port": 80,
        "protocol": 6,    # TCP
        "length": 40,
        "tcp_flags": {"SYN": True},
        "payload": ""
    },
    # ... 更多来自不同源IP的类似数据包
]

# 异常大流量数据包（可能是数据泄露）
data_exfiltration_packets = [
    {
        "timestamp": 1620000300.0,
        "src_ip": "192.168.1.100",
        "dst_ip": "203.0.113.100",
        "src_port": 1234,
        "dst_port": 8080,
        "protocol": 6,    # TCP
        "length": 1500,
        "tcp_flags": {"SYN": True},
        "payload": ""
    },
    {
        "timestamp": 1620000300.1,
        "src_ip": "203.0.113.100",
        "dst_ip": "192.168.1.100",
        "src_port": 8080,
        "dst_port": 1234,
        "protocol": 6,    # TCP
        "length": 1500,
        "tcp_flags": {"SYN": True, "ACK": True},
        "payload": ""
    },
    {
        "timestamp": 1620000300.2,
        "src_ip": "192.168.1.100",
        "dst_ip": "203.0.113.100",
        "src_port": 1234,
        "dst_port": 8080,
        "protocol": 6,    # TCP
        "length": 1500,
        "tcp_flags": {"ACK": True},
        "payload": ""
    },
    {
        "timestamp": 1620000300.3,
        "src_ip": "192.168.1.100",
        "dst_ip": "203.0.113.100",
        "src_port": 1234,
        "dst_port": 8080,
        "protocol": 6,    # TCP
        "length": 1500,
        "tcp_flags": {"ACK": True, "PSH": True},
        "payload": "Large payload data..."  # 实际会是大量数据
    },
    # ... 持续发送大量数据包
]

# 异常流量的特征样本
anomaly_features = [
    # 端口扫描特征
    {
        "packet_count": 6,
        "total_bytes": 360,
        "avg_packet_size": 60.0,
        "std_packet_size": 0.0,
        "min_packet_size": 60,
        "max_packet_size": 60,
        "duration": 0.5,
        "packet_rate": 12.0,
        "byte_rate": 720.0,
        "tcp_syn_count": 6,
        "tcp_ack_count": 0,
        "tcp_fin_count": 0,
        "inter_arrival_mean": 0.1,
        "inter_arrival_std": 0.0,
        "burst_count": 1,
        "port_count": 6,  # 异常多的目标端口
        "protocol": 6
    },
    # DDoS攻击特征
    {
        "packet_count": 1000,
        "total_bytes": 40000,
        "avg_packet_size": 40.0,
        "std_packet_size": 0.0,
        "min_packet_size": 40,
        "max_packet_size": 40,
        "duration": 1.0,
        "packet_rate": 1000.0,  # 异常高的速率
        "byte_rate": 40000.0,
        "tcp_syn_count": 1000,
        "tcp_ack_count": 0,
        "src_ip_count": 1000,  # 异常多的源IP
        "inter_arrival_mean": 0.001,
        "inter_arrival_std": 0.0,
        "burst_count": 1,
        "protocol": 6
    },
    # 数据泄露特征
    {
        "packet_count": 5000,
        "total_bytes": 7500000,  # 7.5MB的流量
        "avg_packet_size": 1500.0,
        "std_packet_size": 0.0,
        "duration": 60.0,
        "packet_rate": 83.3,
        "byte_rate": 125000.0,  # 125KB/s的持续上传
        "direction_ratio": 0.99,  # 几乎全是外向流量
        "protocol": 6
    }
]
