#!/bin/bash

# ============================================================
# AgentEvolver 单机训练流水线 (Single Node Standalone)
# ============================================================

# 获取本机 IP (用于 no_proxy 设置，防止 Python 库报错)
HOST_IP=$(hostname -I | awk '{print $1}')

# ---- 1. 清理旧进程 ----
echo "🧹 Cleaning up previous processes..."
pkill -9 -f agentevolver.main_ppo
pkill -9 -f vllm
pkill -f "env_service.env_service"
# 停止旧的 Ray 进程
ray stop --force >/dev/null 2>&1
# 确保端口空闲
fuser -k -9 8080/tcp >/dev/null 2>&1
fuser -k -9 6379/tcp >/dev/null 2>&1
fuser -k -9 8265/tcp >/dev/null 2>&1

sleep 2

# ---- 2. 网络与稳定性配置 ----
export SETUPTOOLS_USE_DISTUTILS=local
# 代理设置
export http_proxy=http://hk-mmhttpproxy.woa.com:11113
export https_proxy=http://hk-mmhttpproxy.woa.com:11113
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy

# 关键：单机也需要这些配置防止 vLLM 崩溃
export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=bond1
export TP_SOCKET_IFNAME=bond1
# 强制关闭 P2P 和 CUDA Graph，保证最强稳定性
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export VLLM_ENFORCE_EAGER=True
export OMP_NUM_THREADS=1

# 代理绕过配置
export no_proxy="localhost,127.0.0.1,::1,0.0.0.0,29.209.112.175,$HOST_IP,29.0.0.0/8,10.0.0.0/8,172.16.0.0/12,.woa.com"
export NO_PROXY=$no_proxy

echo "✅ Network & Proxy Configured."

# ---- 3. 启动本地 Ray Head ----
# 单机模式下，我们直接在这里启动 Ray，不需要额外的 start_ray_head.sh
echo "🚀 Starting Local Ray Cluster..."
# 限制 Ray 只使用本机资源，防止它尝试连接其他机器
ray start --head --port=6379 --num-gpus=8 --dashboard-host=0.0.0.0 --disable-usage-stats

sleep 5

# ---- 4. 启动 AppWorld 环境服务 ----
if [ -f "./env_service/launch_script/appworld_multi.sh" ]; then
    echo "🌍 Starting AppWorld Service (Conda: appworld)..."
    
    (
        CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate appworld
        
        # 确保数据路径变量存在 (防止 Ray Worker 找不到数据)
        APPWORLD_DIR="$(pwd)/env_service/environments/appworld"
        export APPWORLD_ROOT="$APPWORLD_DIR"
        export PYTHONPATH="$(pwd):$APPWORLD_DIR:$PYTHONPATH"

        # 启动服务
        bash ./env_service/launch_script/appworld_multi.sh > appworld_server.log 2>&1
    ) &
    APPWORLD_PID=$!
    
    echo "Waiting 30s for AppWorld to initialize..."
    sleep 30
else
    echo "❌ Error: appworld_multi.sh not found!"
    ray stop --force
    exit 1
fi

# 退出陷阱：脚本结束时清理 Ray 和 AppWorld
trap "echo '🛑 Stopping Services...'; kill $APPWORLD_PID; ray stop --force" EXIT

# ---- 5. 调用训练脚本 ----
echo "🔥 Starting Training Pipeline (Single Node)..."
bash ./Our_examples/run_api_driven_H20_single.sh