#!/bin/bash

# --- 环境变量设置 (保持不变) ---
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

export http_proxy=http://hk-mmhttpproxy.woa.com:11113
export https_proxy=http://hk-mmhttpproxy.woa.com:11113
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=bond1
export TP_SOCKET_IFNAME=bond1
HOST_IP=$(hostname -I | awk '{print $1}')
export no_proxy="localhost,127.0.0.1,::1,0.0.0.0,29.209.112.175,$HOST_IP,29.0.0.0/8,10.0.0.0/8,172.16.0.0/12,.woa.com"
export NO_PROXY=$no_proxy
export GEN_OUTPUT_DIR="/mnt/cephfs/haowengao/test_agent/GEN_NEW_DATA"
export RAY_worker_register_timeout_seconds=600
export RAY_maximum_startup_concurrency=16

# --- 关键修改：智能检查 Ray 状态 ---

# 检查 Ray 是否已经在这个节点上运行
ray status >/dev/null 2>&1
RAY_STATUS=$?

if [ $RAY_STATUS -eq 0 ]; then
    echo "✅ Ray Cluster is already running. Skipping restart."
    echo "🧹 Cleaning up previous python job processes only..."
    
    # 只杀掉业务相关的 Python 进程，不要杀 Ray 进程
    pkill -f "env_service.env_service"
    
    # 这里的 sleep 可以缩短，因为不需要等待 ray start
    sleep 2
else
    echo "⚠️ Ray is NOT running. Starting fresh Head node..."
    
    # 只有 Ray 没运行时，才执行彻底的清理和重启
    ray stop --force
    pkill -f "env_service.env_service"
    rm -rf /tmp/ray/*

    ray start --head \
        --port=6379 \
        --node-ip-address=29.209.112.175 \
        --num-cpus=64 \
        --num-gpus=8 \
        --dashboard-host=0.0.0.0 \
        --disable-usage-stats
        
    echo "⏳ Waiting for Ray to fully start..."
    sleep 20
fi

# --- 提交任务 ---
echo "🚀 Submitting job..."
bash ./Our_examples/run_multi_node_v1.sh