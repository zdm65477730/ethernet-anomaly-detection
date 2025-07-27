#!/bin/bash
# 异常检测系统依赖安装脚本
# 功能：自动安装系统依赖、Python环境和项目所需包
# 使用方法：chmod +x install_dependencies.sh && sudo ./install_dependencies.sh

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # 无颜色

# 检查是否以root权限运行
if [ "$(id -u)" -ne 0 ]; then
    echo -e "${RED}错误：请使用sudo或root权限运行此脚本${NC}"
    exit 1
fi

# 安装系统依赖
install_system_dependencies() {
    echo -e "${YELLOW}=== 安装系统依赖 ===${NC}"
    
    # 检测操作系统类型
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$NAME
    else
        echo -e "${RED}不支持的操作系统${NC}"
        exit 1
    fi

    # 根据不同系统使用相应的包管理器
    if [ "$OS" = "Ubuntu" ] || [ "$OS" = "Debian GNU/Linux" ]; then
        apt-get update -y
        apt-get install -y \
            python3 \
            python3-pip \
            python3-venv \
            libpcap-dev \
            build-essential \
            libssl-dev \
            libffi-dev \
            python3-dev \
            git \
            curl \
            tar \
            unzip
    elif [ "$OS" = "CentOS Linux" ] || [ "$OS" = "Red Hat Enterprise Linux" ]; then
        yum install -y \
            python3 \
            python3-pip \
            python3-devel \
            libpcap-devel \
            gcc \
            gcc-c++ \
            make \
            openssl-devel \
            libffi-devel \
            git \
            curl \
            tar \
            unzip
    else
        echo -e "${RED}不支持的操作系统：$OS${NC}"
        exit 1
    fi

    echo -e "${GREEN}系统依赖安装完成${NC}"
}

# 创建并配置Python虚拟环境
setup_python_env() {
    echo -e "${YELLOW}=== 配置Python环境 ===${NC}"
    
    # 检查项目根目录是否存在
    if [ ! -d "./src" ] && [ ! -d "../src" ]; then
        echo -e "${RED}错误：未在项目根目录下运行脚本，请确保在项目根目录执行此脚本${NC}"
        exit 1
    fi
    
    # 确定项目根目录路径
    if [ -d "./src" ]; then
        PROJECT_ROOT="."
    else
        PROJECT_ROOT=".."
    fi
    
    # 创建虚拟环境
    if [ ! -d "$PROJECT_ROOT/venv" ]; then
        python3 -m venv $PROJECT_ROOT/venv
        echo -e "${GREEN}虚拟环境创建成功${NC}"
    else
        echo -e "${YELLOW}虚拟环境已存在，跳过创建${NC}"
    fi
    
    # 激活虚拟环境并安装依赖
    echo -e "${YELLOW}安装Python依赖包...${NC}"
    source $PROJECT_ROOT/venv/bin/activate
    
    # 升级pip
    pip install --upgrade pip
    
    # 安装基础依赖
    # https://mirrors.aliyun.com/pypi/simple/ or https://pypi.tuna.tsinghua.edu.cn/simple
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        pip install -r $PROJECT_ROOT/requirements.txt -i https://mirrors.aliyun.com/pypi/simple
    else
        echo -e "${RED}错误：未找到requirements.txt${NC}"
        exit 1
    fi
    
    # 询问是否安装可选的GPU支持
    read -p "是否安装GPU支持的依赖？(y/n，需要NVIDIA显卡) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ -f "$PROJECT_ROOT/requirements_gpu.txt" ]; then
            pip install -r $PROJECT_ROOT/requirements_gpu.txt -i https://mirrors.aliyun.com/pypi/simple
            echo -e "${GREEN}GPU依赖安装完成${NC}"
        else
            echo -e "${YELLOW}未找到GPU依赖文件，跳过${NC}"
        fi
    fi
    
    deactivate
    echo -e "${GREEN}Python依赖安装完成${NC}"
}

# 主函数
main() {
    echo -e "${YELLOW}===== 异常检测系统依赖安装工具 =====${NC}"
    install_system_dependencies
    setup_python_env
    echo -e "${GREEN}===== 所有依赖安装完成 =====${NC}"
    echo -e "${YELLOW}使用说明：${NC}"
    echo -e "1. 激活虚拟环境：source venv/bin/activate"
    echo -e "2. 启动系统：anomaly-detector start"
}

# 执行主函数
main