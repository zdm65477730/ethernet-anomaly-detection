"""正常网络流量样本数据"""

# 正常HTTP会话数据包（3个包）
normal_http_packets = [
    {
        "timestamp": 1620000000.0,
        "src_ip": "192.168.1.100",
        "dst_ip": "203.0.113.5",
        "src_port": 45678,
        "dst_port": 80,
        "protocol": 6,  # TCP
        "length": 78,
        "tcp_flags": {"SYN": True},
        "payload": ""
    },
    {
        "timestamp": 1620000000.2,
        "src_ip": "203.0.113.5",
        "dst_ip": "192.168.1.100",
        "src_port": 80,
        "dst_port": 45678,
        "protocol": 6,  # TCP
        "length": 78,
        "tcp_flags": {"SYN": True, "ACK": True},
        "payload": ""
    },
    {
        "timestamp": 1620000000.3,
        "src_ip": "192.168.1.100",
        "dst_ip": "203.0.113.5",
        "src_port": 45678,
        "dst_port": 80,
        "protocol": 6,  # TCP
        "length": 66,
        "tcp_flags": {"ACK": True},
        "payload": ""
    },
    {
        "timestamp": 1620000000.4,
        "src_ip": "192.168.1.100",
        "dst_ip": "203.0.113.5",
        "src_port": 45678,
        "dst_port": 80,
        "protocol": 6,  # TCP
        "length": 512,
        "tcp_flags": {"ACK": True, "PSH": True},
        "payload": "GET /index.html HTTP/1.1\r\nHost: example.com\r\n..."
    },
    {
        "timestamp": 1620000000.7,
        "src_ip": "203.0.113.5",
        "dst_ip": "192.168.1.100",
        "src_port": 80,
        "dst_port": 45678,
        "protocol": 6,  # TCP
        "length": 1500,
        "tcp_flags": {"ACK": True, "PSH": True},
        "payload": "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n...<html>..."
    },
    {
        "timestamp": 1620000001.0,
        "src_ip": "192.168.1.100",
        "dst_ip": "203.0.113.5",
        "src_port": 45678,
        "dst_port": 80,
        "protocol": 6,  # TCP
        "length": 66,
        "tcp_flags": {"ACK": True},
        "payload": ""
    },
    {
        "timestamp": 1620000005.0,
        "src_ip": "192.168.1.100",
        "dst_ip": "203.0.113.5",
        "src_port": 45678,
        "dst_port": 80,
        "protocol": 6,  # TCP
        "length": 66,
        "tcp_flags": {"FIN": True, "ACK": True},
        "payload": ""
    }
]

# 正常DNS查询数据包
normal_dns_packets = [
    {
        "timestamp": 1620000010.0,
        "src_ip": "192.168.1.100",
        "dst_ip": "198.51.100.1",
        "src_port": 53452,
        "dst_port": 53,
        "protocol": 17,  # UDP
        "length": 54,
        "payload": "DNS Query: example.com A"
    },
    {
        "timestamp": 1620000010.1,
        "src_ip": "198.51.100.1",
        "dst_ip": "192.168.1.100",
        "src_port": 53,
        "dst_port": 53452,
        "protocol": 17,  # UDP
        "length": 70,
        "payload": "DNS Response: example.com A 203.0.113.5"
    }
]

# 正常SSH会话数据包
normal_ssh_packets = [
    {
        "timestamp": 1620000020.0,
        "src_ip": "192.168.1.100",
        "dst_ip": "198.51.100.10",
        "src_port": 38721,
        "dst_port": 22,
        "protocol": 6,  # TCP
        "length": 78,
        "tcp_flags": {"SYN": True},
        "payload": ""
    },
    {
        "timestamp": 1620000020.1,
        "src_ip": "198.51.100.10",
        "dst_ip": "192.168.1.100",
        "src_port": 22,
        "dst_port": 38721,
        "protocol": 6,  # TCP
        "length": 78,
        "tcp_flags": {"SYN": True, "ACK": True},
        "payload": ""
    },
    # 更多SSH握手和数据传输包...
]

# 正常流量的特征样本
normal_features = [
    # HTTP会话特征
    {
        "packet_count": 7,
        "total_bytes": 2366,
        "avg_packet_size": 338.0,
        "std_packet_size": 542.3,
        "min_packet_size": 66,
        "max_packet_size": 1500,
        "duration": 5.0,
        "packet_rate": 1.4,
        "byte_rate": 473.2,
        "tcp_syn_count": 2,
        "tcp_ack_count": 6,
        "tcp_fin_count": 1,
        "inter_arrival_mean": 0.83,
        "inter_arrival_std": 1.67,
        "burst_count": 0,
        "protocol": 6
    },
    # DNS会话特征
    {
        "packet_count": 2,
        "total_bytes": 124,
        "avg_packet_size": 62.0,
        "std_packet_size": 11.3,
        "min_packet_size": 54,
        "max_packet_size": 70,
        "duration": 0.1,
        "packet_rate": 20.0,
        "byte_rate": 1240.0,
        "inter_arrival_mean": 0.1,
        "inter_arrival_std": 0.0,
        "burst_count": 0,
        "protocol": 17
    }
]
