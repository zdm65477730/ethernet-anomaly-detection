#!/bin/bash
# 日志清理脚本
# 功能：清理超过指定天数的日志文件，支持预览模式
# 使用方法：chmod +x clear_old_logs.sh && ./clear_old_logs.sh [--days 天数] [--preview]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 默认配置
DEFAULT_LOG_DIR="../logs"  # 默认日志目录
DEFAULT_DAYS=30            # 默认保留天数
PREVIEW_MODE=false         # 预览模式（不实际删除）

# 解析命令行参数
LOG_DIR=$DEFAULT_LOG_DIR
DAYS=$DEFAULT_DAYS

while [[ $# -gt 0 ]]; do
    case "$1" in
        --days)
            DAYS="$2"
            shift 2
            ;;
        --preview)
            PREVIEW_MODE=true
            shift
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}错误：未知参数 $1${NC}"
            echo "使用方法：$0 [--days 天数] [--preview] [--log-dir 日志目录]"
            exit 1
            ;;
    esac
done

# 验证参数
if ! [[ "$DAYS" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}错误：天数必须是正整数${NC}"
    exit 1
fi

# 检查日志目录是否存在
if [ ! -d "$LOG_DIR" ]; then
    echo -e "${RED}错误：日志目录 $LOG_DIR 不存在${NC}"
    exit 1
fi

# 查找过期日志文件
echo -e "${YELLOW}正在查找 $LOG_DIR 下超过 $DAYS 天的日志文件...${NC}"
OLD_LOGS=$(find "$LOG_DIR" -type f -name "*.log*" -mtime +"$DAYS")

# 统计文件数量和大小
if [ -z "$OLD_LOGS" ]; then
    echo -e "${GREEN}未找到符合条件的过期日志文件${NC}"
    exit 0
fi

FILE_COUNT=$(echo "$OLD_LOGS" | wc -l | tr -d ' ')
TOTAL_SIZE=$(du -ch $OLD_LOGS | grep total | cut -f1)

# 显示预览信息
echo -e "${BLUE}找到 $FILE_COUNT 个过期日志文件，总大小：$TOTAL_SIZE${NC}"
echo -e "${BLUE}文件列表：${NC}"
echo "$OLD_LOGS" | while read -r logfile; do
    echo "  - $logfile ($(du -h "$logfile" | cut -f1))"
done

# 执行删除（非预览模式）
if [ "$PREVIEW_MODE" = false ]; then
    read -p "确定要删除这些文件吗？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}正在删除文件...${NC}"
        echo "$OLD_LOGS" | xargs rm -f
        
        # 验证删除结果
        REMAINING_LOGS=$(find "$LOG_DIR" -type f -name "*.log*" -mtime +"$DAYS")
        if [ -z "$REMAINING_LOGS" ]; then
            echo -e "${GREEN}所有过期日志文件已成功删除${NC}"
        else
            echo -e "${YELLOW}部分文件删除失败，剩余文件：${NC}"
            echo "$REMAINING_LOGS"
        fi
    else
        echo -e "${YELLOW}已取消删除操作${NC}"
    fi
else
    echo -e "${YELLOW}预览模式：未执行删除操作${NC}"
fi

echo -e "${GREEN}日志清理任务完成${NC}"
