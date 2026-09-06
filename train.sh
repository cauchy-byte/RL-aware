#!/bin/bash

# 固定指定使用GPU1（空闲卡）
export CUDA_VISIBLE_DEVICES=3

PYTHON_BIN="python3"
if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
fi

# 环境选择：Ant-v4
# 其他可选环境：HalfCheetah-v4, Hopper-v4, Walker2d-v4, Humanoid-v4
ENV_NAME="Walker2d-v4"

MIN_BATCH_SIZE=1000

# EP输入状态序列长度H（每次用最近H个state，每个state对齐一个动作窗口）
EP_STATE_HISTORY_LENGTH=32
# EP输入动作窗口长度T（每个state对应最近T个动作的滑动窗口）
EP_ACTION_WINDOW_LENGTH=10
# TBPTT / RNN序列长度，建议与EP_STATE_HISTORY_LENGTH一致
RNN_FIX_LENGTH=$EP_STATE_HISTORY_LENGTH
# 兼容旧参数名：action_history_length 用作动作窗口长度T
ACTION_HISTORY_LENGTH=$EP_ACTION_WINDOW_LENGTH

# 每个episode是否随机延迟参数（默认开启）
RANDOM_DELAY_PER_EPISODE=1  # 1=开启随机, 0=关闭随机
DELAY_PROCESS="${DELAY_PROCESS:-ge1_23}"
DELAY_TASK_STRATEGY="${DELAY_TASK_STRATEGY:-acda_cycle}"
DELAY_TASK_NUM="${DELAY_TASK_NUM:-3}"
TRAINING_WORKER_NUM="${TRAINING_WORKER_NUM:-3}"

echo "=================================================="
echo "延迟环境训练配置"
echo "=================================================="
echo "  - 环境名称: $ENV_NAME"
echo "  - min_batch_size: $MIN_BATCH_SIZE (三个训练 worker 合计 transition 目标)"
echo "  - update_interval: 10"
echo "  - 使用延迟环境: True"
echo "  - 非平稳延迟: True"
echo "  - 初始延迟类型: gamma"
echo "  - 最大延迟范围: 3-10"
echo "  - 延迟改变周期: 10000 steps"
echo "  - 延迟改变间隔: 500 steps"
echo "  - 训练 worker 数: $TRAINING_WORKER_NUM"
echo "  - 表征学习: RMDM + contrastive, WMCL disabled"
echo "  - EP_STATE_HISTORY_LENGTH(H): $EP_STATE_HISTORY_LENGTH"
echo "  - EP_ACTION_WINDOW_LENGTH(T): $EP_ACTION_WINDOW_LENGTH"
echo "  - rnn_fix_length: $RNN_FIX_LENGTH"
echo "  - action_history_length(compat=T): $ACTION_HISTORY_LENGTH"
echo "  - 每个Episode随机延迟: $RANDOM_DELAY_PER_EPISODE"
echo "  - 延迟过程: $DELAY_PROCESS"
echo "  - 延迟任务策略: $DELAY_TASK_STRATEGY"
echo "  - 延迟任务数: $DELAY_TASK_NUM"
echo ""
echo "=================================================="
echo ""

${PYTHON_BIN} main.py \
    --env_name $ENV_NAME \
    --minimal_repre_rp_size 0 \
    --use_rmdm  \
    --bottle_neck \
    --use_contrastive  \
    --stop_pg_for_ep  \
    --rmdm_update_interval 10 \
    --rbf_radius 3000.0 \
    --min_batch_size $MIN_BATCH_SIZE \
    --delay_changing_period 10000 \
    --delay_changing_interval 500 \
    --delay_process "$DELAY_PROCESS" \
    --delay_task_strategy "$DELAY_TASK_STRATEGY" \
    --delay_task_num "$DELAY_TASK_NUM" \
    --training_worker_num "$TRAINING_WORKER_NUM" \
    --update_interval 10 \
    --value_learning_rate 0.0003 \
    --target_entropy_ratio 1.0 \
    --alpha_min 0.02 \
    --max_iter_num 3000 \
    --action_history_length $ACTION_HISTORY_LENGTH \
    --ep_state_history_length $EP_STATE_HISTORY_LENGTH \
    --ep_action_window_length $EP_ACTION_WINDOW_LENGTH \
    --rnn_fix_length $RNN_FIX_LENGTH \
    --random_delay_per_episode $RANDOM_DELAY_PER_EPISODE

echo ""
echo "=================================================="
echo "训练完成或已停止"
echo "=================================================="
