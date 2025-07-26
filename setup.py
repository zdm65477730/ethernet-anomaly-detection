from setuptools import setup, find_packages
import os

def read_file(filename):
    """读取文件内容"""
    with open(os.path.join(os.path.dirname(__file__), filename), 'r', encoding='utf-8') as f:
        return f.read()

setup(
    name="anomaly-detection-system",
    version="1.0.0",
    author="Security Research Team",
    author_email="security@example.com",
    description="实时网络异常检测系统，基于机器学习的网络安全防护工具",
    long_description=read_file('README.md'),
    long_description_content_type="text/markdown",
    url="https://github.com/example/anomaly-detection-system",
    packages=find_packages(exclude=["tests", "tests.*", "docs", "scripts"]),
    package_data={
        "src.config": ["*.yaml", "*.conf"],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "anomaly-detector = src.cli:main",
            "ads = src.cli:main",  # 简写命令
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: System Administrators",
        "Intended Audience :: Security Professionals",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Security",
        "Topic :: System :: Networking :: Monitoring",
    ],
    python_requires=">=3.8",
    install_requires=read_file('requirements.txt').splitlines(),
    project_urls={
        "Documentation": "https://example.com/docs/anomaly-detection",
        "Source": "https://github.com/example/anomaly-detection-system",
        "Tracker": "https://github.com/example/anomaly-detection-system/issues",
    },
)