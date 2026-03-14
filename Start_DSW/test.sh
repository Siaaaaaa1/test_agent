#!/bin/bash

# 1. 遇到任何错误立即停止脚本运行
set -e

# 2. 进入项目主目录
cd /mnt/workspace/haowengao/AgentEvolver

# 3. 初始化并激活 Conda 环境
# 使用绝对路径 source 激活脚本，这是在 DSW 或服务器脚本中最稳定的方式
source /mnt/workspace/haowengao/miniconda3/bin/activate agentevolver

# 4. 启动训练脚本
# 这里使用 bash 明确调用，并指向你指定的长路径脚本
bash ./Ablation_AE_Components/0-full.sh