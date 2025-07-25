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
        mock_raw_packet = b'\x45\x00\x00\x3c\x1c\x46\x40\x00\x40\x06\x00\x00\xc0\xa8\x01\x01\xc0\xa8\x01\x02\x04\xd2\x00\x50\x00\x00\x00\x00\x00\x00\x00\x00\x50\x02\x20\x00\x7d\x32\x00\x00'
        
        # 解析数据包
        parsed = self.capture._parse_packet(mock_raw_packet, 1620000000.123456)
        
        # 验证解析结果
        self.assertEqual(parsed['src_ip'], '192.168.1.1')
        self.assertEqual(parsed['dst_ip'], '192.168.1.2')
        self.assertEqual(parsed['src_port'], 1234)
        self.assertEqual(parsed['dst_port'], 80)
        self.assertEqual(parsed['protocol'], 6)  # TCP
        self.assertEqual(parsed['length'], 60)

class TestSessionTracker(unittest.TestCase):
    """测试会话跟踪功能"""
    
    def setUp(self):
        """初始化测试环境"""
        self.tracker = SessionTracker(timeout=10)  # 短超时便于测试
        self.sample_packet = {
            "timestamp": 1620000000.123456,
            "src_ip": "192.168.1.100",
            "dst_ip": "8.8.8.8",
            "src_port": 12345,
            "dst_port": 80,
            "protocol": 6,
            "length": 150
        }
    
    def test_session_creation(self):
        """测试会话创建功能"""
        session_id, session = self.tracker.track_packet(self.sample_packet)
        
        # 验证会话ID格式和会话创建
        self.assertIsNotNone(session_id)
        self.assertEqual(len(session_id), 32)  # 假设是32位哈希
        self.assertEqual(session['packet_count'], 1)
        self.assertEqual(session['total_bytes'], 150)
        self.assertEqual(session['src_ip'], "192.168.1.100")
        self.assertEqual(session['dst_ip'], "8.8.8.8")
    
    def test_session_update(self):
        """测试会话更新功能"""
        # 添加第一个包
        session_id, _ = self.tracker.track_packet(self.sample_packet)
        
        # 添加第二个包
        second_packet = self.sample_packet.copy()
        second_packet['length'] = 200
        second_packet['timestamp'] = 1620000001.123456
        _, session = self.tracker.track_packet(second_packet)
        
        # 验证会话已更新
        self.assertEqual(session['packet_count'], 2)
        self.assertEqual(session['total_bytes'], 350)
        self.assertAlmostEqual(session['duration'], 1.0, places=3)
    
    def test_session_expiry(self):
        """测试会话过期清理功能"""
        # 创建会话
        session_id, _ = self.tracker.track_packet(self.sample_packet)
        self.assertEqual(len(self.tracker.sessions), 1)
        
        # 模拟时间过了超时时间
        expired_packet = self.sample_packet.copy()
        expired_packet['timestamp'] = 1620000000.123456 + 15  # 超过10秒超时
        self.tracker.track_packet(expired_packet)
        
        # 清理过期会话
        cleaned = self.tracker.cleanup_expired()
        self.assertEqual(cleaned, 1)
        self.assertEqual(len(self.tracker.sessions), 1)  # 新会话应该保留
        self.assertEqual(len(self.tracker.sessions), 1)  # 新会话应该保留

if __name__ == '__main__':
    unittest.main()
