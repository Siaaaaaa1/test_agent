#!/bin/bash

# ============================================================
# AgentEvolver 多机版 - 流程控制脚本 (Master 专用)
# 运行此脚本前，请确保你已经运行了 Ray Head 和 Ray Worker 脚本
# ============================================================

# ---- 1. 清理本地环境服务 (不清理 Ray) ----
echo "🧹 Cleaning up previous environment services..."
# 注意：这里只杀环境服务和主进程，不执行 ray stop，以免影响你手动起的集群
pkill -f "env_service.env_service"
fuser -k -9 8080/tcp >/dev/null 2>&1

# ---- 2. 网络与代理配置 (与你的 Ray 脚本保持 100% 一致) ----
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export http_proxy=http://hk-mmhttpproxy.woa.com:11113
export https_proxy=http://hk-mmhttpproxy.woa.com:11113
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=bond1
export TP_SOCKET_IFNAME=bond1
export GEN_OUTPUT_DIR="/mnt/cephfs/haowengao/test_agent/GEN_NEW_DATA"

MASTER_IP="29.209.112.175"
HOST_IP=$(hostname -I | awk '{print $1}')
export no_proxy="localhost,127.0.0.1,::1,0.0.0.0,$MASTER_IP,$HOST_IP,29.0.0.0/8,10.0.0.0/8,172.16.0.0/12,.woa.com"
export NO_PROXY=$no_proxy

# ---- 3. 检查 Ray 集群状态 ----
echo "🔍 Checking Ray Cluster..."
if ! ray status >/dev/null 2>&1; then
    echo "❌ Ray cluster is not running! Please run your Ray Head/Worker scripts first."
    exit 1
fi
echo "✅ Ray Cluster is alive."

# ---- 4. 启动 AppWorld 服务 ----
echo "🌍 Launching AppWorld Service..."
LAUNCHER_SCRIPT="./env_service/launch_script/appworld_multi_v2.sh"
chmod +x "$LAUNCHER_SCRIPT"

# 启动服务并将日志记录到 server.log
$LAUNCHER_SCRIPT 2>&1 | tee server.log &
SERVER_PID=$!

# 退出陷阱：只清理环境服务，不关 Ray
trap "echo '🛑 Stopping Environment Service...'; kill $SERVER_PID; exit" EXIT

# ---- 5. 环境健康检查 ----
echo "⏳ Waiting for AppWorld Service at http://${MASTER_IP}:8080..."
COUNT=0
while ! curl -s --noproxy "*" "http://${MASTER_IP}:8080/healthz" > /dev/null; do
    sleep 1
    COUNT=$((COUNT+1))
    if ! kill -0 $SERVER_PID 2>/dev/null; then echo "❌ Service died!"; exit 1; fi
    if [ $COUNT -ge 300 ]; then echo "❌ Timeout!"; exit 1; fi
    echo -ne "   Waiting... ${COUNT}s\r"
done
echo -e "\n✅ AppWorld is UP!"

# ---- 6. 执行训练 ----
bash ./Our_examples/run_ppo_multi_core.sh