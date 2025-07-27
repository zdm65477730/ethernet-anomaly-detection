"""命令行接口入口模块"""
import sys
from src.cli.main import app

if __name__ == "__main__":
    app(prog_name="anomaly-detector")