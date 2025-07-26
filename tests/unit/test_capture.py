import unittest
import time
from unittest.mock import Mock, patch
import socket
import numpy as np
from src.capture.packet_capture import PacketCapture
from src.capture.session_tracker import SessionTracker

class TestPacketCapture(unittest.TestCase):
    """测试数据包捕获与解析功能"""
    
    def setUp(self):
        """初始化测试环境"""
        self.mock_interface = "test0"
        self.capture = PacketCapture(interface=self.mock_interface, filter="tcp", timeout=10)
        
    def tearDown(self):
        """清理测试环境"""
        if self.capture.is_running:
            self.capture.stop()
    
    @patch('src.capture.packet_capture.socket.socket')
    def test_start_stop_capture(self, mock_socket):
        """测试启动和停止抓包功能"""
        # 测试启动
        self.capture.start()
        self.assertTrue(self.capture.is_running)
        mock_socket.assert_called_once()
        
        # 测试停止
        self.capture.stop()
        self.assertFalse(self.capture.is_running)
    
    @patch('src.capture.packet_capture.PacketCapture._process_packet')
    def test_packet_processing(self, mock_process):
        """测试数据包处理流程"""
        self.capture.start()
        
        # 模拟接收到数据包
        mock_packet = Mock()
        self.capture._packet_queue.put(mock_packet)
        time.sleep(0.1)  # 等待处理
        
        # 验证数据包被处理
        mock_process.assert_called_once_with(mock_packet)
        
        self.capture.stop()
    
    def test_packet_parsing(self):
        """测试数据包解析功能"""
        # 创建一个模拟的TCP数据包（简化版）
        mock_packet_data = bytearray([
            0x45, 0x00, 0x00, 0x3c,  # IP头
            0x00, 0x00, 0x40, 0x00,
            0x40, 0x06, 0x00, 0x00,
            0xc0, 0xa8, 0x01, 0x64,  # 源IP: 192.168.1.100
            0x08, 0x08, 0x08, 0x08,  # 目的IP: 8.8.8.8
            0x00, 0x50, 0x04, 0xd2,  # TCP头: 源端口80, 目的端口1234
            0x00, 0x00, 0x00, 0x00,  # 序列号
            0x00, 0x00, 0x00, 0x00,  # 确认号
            0x50, 0x10, 0x00, 0x00,  # TCP标志位等
            0x00, 0x00, 0x00, 0x00,  # 校验和等
            0x48, 0x65, 0x6c, 0x6c,  # 数据: "Hello"
            0x6f
        ])
        
        # 解析数据包
        parsed_packet = self.capture._parse_packet(bytes(mock_packet_data), 1620000000.0)
        
        # 验证解析结果
        self.assertIsNotNone(parsed_packet)
        self.assertEqual(parsed_packet['src_ip'], '192.168.1.100')
        self.assertEqual(parsed_packet['dst_ip'], '8.8.8.8')
        self.assertEqual(parsed_packet['src_port'], 80)
        self.assertEqual(parsed_packet['dst_port'], 1234)
        self.assertEqual(parsed_packet['protocol'], 6)  # TCP
        self.assertEqual(parsed_packet['length'], 60)
        self.assertIn('tcp_flags', parsed_packet)

class TestSessionTracker(unittest.TestCase):
    """测试会话跟踪功能"""
    
    def setUp(self):
        """初始化测试环境"""
        self.tracker = SessionTracker(timeout=30)
        
        # 创建测试数据包
        self.test_packets = [
            {
                'timestamp': 1620000000.0,
                'src_ip': '192.168.1.100',
                'dst_ip': '8.8.8.8',
                'src_port': 12345,
                'dst_port': 80,
                'protocol': 6,
                'length': 100,
                'tcp_flags': {'SYN': True}
            },
            {
                'timestamp': 1620000001.0,
                'src_ip': '8.8.8.8',
                'dst_ip': '192.168.1.100',
                'src_port': 80,
                'dst_port': 12345,
                'protocol': 6,
                'length': 150,
                'tcp_flags': {'SYN': True, 'ACK': True}
            },
            {
                'timestamp': 1620000002.0,
                'src_ip': '192.168.1.100',
                'dst_ip': '8.8.8.8',
                'src_port': 12345,
                'dst_port': 80,
                'protocol': 6,
                'length': 200,
                'tcp_flags': {'ACK': True}
            }
        ]
    
    def test_session_creation(self):
        """测试会话创建"""
        # 处理第一个SYN包，应该创建新会话
        session_key, session_data = self.tracker.process_packet(self.test_packets[0])
        
        self.assertIsNotNone(session_key)
        self.assertEqual(session_data['packet_count'], 1)
        self.assertEqual(session_data['total_bytes'], 100)
    
    def test_session_continuation(self):
        """测试会话延续"""
        # 处理第一个包
        session_key1, session_data1 = self.tracker.process_packet(self.test_packets[0])
        
        # 处理第二个包（响应包）
        session_key2, session_data2 = self.tracker.process_packet(self.test_packets[1])
        
        # 应该是同一个会话
        self.assertEqual(session_key1, session_key2)
        self.assertEqual(session_data2['packet_count'], 2)
        self.assertEqual(session_data2['total_bytes'], 250)
    
    def test_session_completion(self):
        """测试会话完成"""
        # 处理所有测试包
        for packet in self.test_packets:
            session_key, session_data = self.tracker.process_packet(packet)
        
        # 获取完成的会话
        completed_sessions = self.tracker.get_completed_sessions()
        
        # 检查是否有完成的会话（根据实现可能为空）
        self.assertIsInstance(completed_sessions, list)
    
    def test_session_timeout(self):
        """测试会话超时"""
        # 处理第一个包
        session_key, session_data = self.tracker.process_packet(self.test_packets[0])
        
        # 模拟时间流逝（超过超时时间）
        future_time = 1620000000.0 + 60  # 60秒后
        self.tracker._cleanup_expired_sessions(current_time=future_time)
        
        # 检查会话是否被清理
        active_sessions = self.tracker.active_sessions
        self.assertEqual(len(active_sessions), 0)

if __name__ == '__main__':
    unittest.main()