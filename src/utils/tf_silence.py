"""
TensorFlow 日志静默工具
在程序最早期阶段导入此模块以屏蔽 TensorFlow 的 CUDA 相关警告和错误
"""

import os
import sys
import logging

def silence_tensorflow():
    """
    静默 TensorFlow 的警告和错误信息
    需要在任何 TensorFlow 导入之前调用
    """
    # 设置环境变量
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # 设置 Python 日志级别
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('absl').setLevel(logging.ERROR)
    
    # 如果已经导入了 tensorflow，则直接设置其日志级别
    if 'tensorflow' in sys.modules:
        import tensorflow as tf
        tf.get_logger().setLevel(logging.ERROR)

def aggressive_silence():
    """
    更加激进的日志屏蔽方法
    直接重定向 stderr 来屏蔽特定的日志信息
    """
    # 设置环境变量
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # 设置 Python 日志级别
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('absl').setLevel(logging.ERROR)
    
    # 保存原始 stderr
    if not hasattr(sys.stderr, '__original_stderr__'):
        original_stderr = sys.stderr
        sys.stderr.__original_stderr__ = original_stderr
        
        class TensorFlowLogFilter:
            def __init__(self, original_stderr):
                self.original_stderr = original_stderr
                self.skip_patterns = [
                    'tensorflow',
                    'cuda', 
                    'cudnn',
                    'cufft',
                    'cublas',
                    'absl',
                    'computation_placer',
                    'oneDNN',
                    'stream_executor',
                    'external/local_xla'
                ]
            
            def write(self, msg):
                # 检查是否包含需要屏蔽的模式
                msg_lower = msg.lower()
                for pattern in self.skip_patterns:
                    if pattern in msg_lower:
                        return  # 直接返回，不写入stderr
                # 如果不包含屏蔽模式，则正常输出
                self.original_stderr.write(msg)
            
            def flush(self):
                self.original_stderr.flush()
        
        # 应用过滤器
        sys.stderr = TensorFlowLogFilter(original_stderr)

# 立即执行静默操作
silence_tensorflow()