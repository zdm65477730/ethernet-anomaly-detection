#!/bin/bash
# 数据备份脚本
# 功能：备份data/目录到带时间戳的压缩文件，支持指定备份路径和保留天数
# 使用方法：chmod +x backup_data.sh && ./backup_data.sh [--path 备份路径] [--keep 保留天数]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 默认配置
DEFAULT_BACKUP_DIR="../backups"  # 默认备份目录
DEFAULT_KEEP_DAYS=30             # 默认保留天数
DATA_DIR="../data"               # 要备份的数据目录

# 解析命令行参数
BACKUP_DIR=$DEFAULT_BACKUP_DIR
KEEP_DAYS=$DEFAULT_KEEP_DAYS

while [[ $# -gt 0 ]]; do
    case "$1" in
        --path)
            BACKUP_DIR="$2"
            shift 2
            ;;
        --keep)
            KEEP_DAYS="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}错误：未知参数 $1${NC}"
            echo "使用方法：$0 [--path 备份路径] [--keep 保留天数]"
            exit 1
            ;;
    esac
done

# 检查数据目录是否存在
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}错误：数据目录 $DATA_DIR 不存在${NC}"
    exit 1
fi

# 创建备份目录
mkdir -p "$BACKUP_DIR"
if [ $? -ne 0 ]; then
    echo -e "${RED}错误：无法创建备份目录 $BACKUP_DIR${NC}"
    exit 1
fi

# 生成带时间戳的备份文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILENAME="data_backup_$TIMESTAMP.tar.gz"
BACKUP_PATH="$BACKUP_DIR/$BACKUP_FILENAME"

# 执行备份
echo -e "${YELLOW}正在备份数据到 $BACKUP_PATH...${NC}"
tar -czf "$BACKUP_PATH" -C "$(dirname "$DATA_DIR")" "$(basename "$DATA_DIR")"

# 检查备份是否成功
if [ -f "$BACKUP_PATH" ] && [ -s "$BACKUP_PATH" ]; then
    echo -e "${GREEN}备份成功，文件大小：$(du -h "$BACKUP_PATH" | cut -f1)${NC}"
else
    echo -e "${RED}备份失败${NC}"
    exit 1
fi

# 清理旧备份
echo -e "${YELLOW}清理 $KEEP_DAYS 天前的旧备份...${NC}"
find "$BACKUP_DIR" -name "data_backup_*.tar.gz" -type f -mtime +"$KEEP_DAYS" -delete

echo -e "${GREEN}备份任务完成${NC}"
