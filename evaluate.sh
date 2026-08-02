#!/bin/bash

python evaluate_delay_shift_escp.py \
    --run-dir "/home/zhangboyuan/ESCP-master_extended/ESCP-master_extended/log_file/Ant-v4-use_rmdm-rnn_len_16-bottle_neck-ep_dim_16-1_N" \
    --initial-delay-type gamma \
    --shifted-delay-type doublegaussian \
    --type-max-delay '{"gamma": 6, "uniform": 7, "doublegaussian": 9}' \
    --episodes 20 \
    --shift-step 40 \
    --post-shift-steps 100