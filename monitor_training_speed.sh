#!/bin/bash
# 训练速度监控脚本
# 用法: ./monitor_training_speed.sh [log_dir]
# 示例: ./monitor_training_speed.sh log_file/Hopper-v2-ep_dim_2-1_N

LOG_DIR=${1:-"log_file/Hopper-v2-ep_dim_2-1_N"}
LOG_FILE="$LOG_DIR/log.txt"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ 错误: 日志文件不存在: $LOG_FILE"
    exit 1
fi

echo "=================================================="
echo "📊 训练速度监控"
echo "=================================================="
echo "日志文件: $LOG_FILE"
echo ""

# 提取最近10次迭代的TrainingPeriod
echo "最近10次迭代的训练时间（秒）："
echo "----------------------------------------"
grep "TrainingPeriod" "$LOG_FILE" | tail -10 | nl -v 1 -w 2 -s '. '

echo ""
echo "----------------------------------------"

# 计算平均时间
AVG_TIME=$(grep "TrainingPeriod" "$LOG_FILE" | tail -10 | awk -F'|' '{sum += $3; count++} END {if (count > 0) print sum/count; else print 0}')

echo "最近10次平均训练时间: ${AVG_TIME}秒"

# 计算预估完成时间
TOTAL_ITERS=$(grep "max_iter_num" "$LOG_DIR/parameter.txt" 2>/dev/null | awk -F': ' '{print $2}' || echo "1000")
COMPLETED_ITERS=$(grep -c "Starting iteration" "$LOG_FILE")
REMAINING_ITERS=$((TOTAL_ITERS - COMPLETED_ITERS))

echo ""
echo "训练进度:"
echo "  - 总迭代次数: $TOTAL_ITERS"
echo "  - 已完成: $COMPLETED_ITERS"
echo "  - 剩余: $REMAINING_ITERS"

if [ $(echo "$AVG_TIME > 0" | bc -l 2>/dev/null || echo "0") -eq 1 ]; then
    REMAINING_SECONDS=$(echo "$AVG_TIME * $REMAINING_ITERS" | bc -l)
    REMAINING_HOURS=$(echo "$REMAINING_SECONDS / 3600" | bc -l)
    REMAINING_MINUTES=$(echo "$REMAINING_SECONDS / 60" | bc -l)
    
    echo ""
    printf "预估剩余时间: %.1f 小时 (%.0f 分钟)\n" "$REMAINING_HOURS" "$REMAINING_MINUTES"
fi

# 检查是否有优化空间
echo ""
echo "性能评估:"
if [ $(echo "$AVG_TIME > 100" | bc -l 2>/dev/null || echo "0") -eq 1 ]; then
    echo "⚠️  迭代时间较长（>${AVG_TIME}秒），建议查看优化指南："
    echo "   cat TRAINING_SPEED_OPTIMIZATION_GUIDE.md"
elif [ $(echo "$AVG_TIME > 50" | bc -l 2>/dev/null || echo "0") -eq 1 ]; then
    echo "ℹ️  迭代时间适中（~${AVG_TIME}秒），可考虑进一步优化"
else
    echo "✅ 迭代时间良好（~${AVG_TIME}秒）"
fi

echo ""
echo "=================================================="
echo "实时监控模式（Ctrl+C退出）"
echo "=================================================="

# 实时监控新的迭代
tail -f "$LOG_FILE" | grep --line-buffered "TrainingPeriod\|Starting iteration" | while read line; do
    if [[ $line == *"Starting iteration"* ]]; then
        ITER=$(echo "$line" | grep -oP 'iteration \K\d+')
        echo ""
        echo "[$(date '+%H:%M:%S')] 开始迭代 $ITER..."
    elif [[ $line == *"TrainingPeriod"* ]]; then
        TIME=$(echo "$line" | awk -F'|' '{print $3}' | xargs)
        echo "[$(date '+%H:%M:%S')] ✓ 完成，耗时: ${TIME}秒"
    fi
done










