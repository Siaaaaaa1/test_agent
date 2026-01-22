#!/bin/bash

# ---- 1. 安全清理 (只杀 Python 任务，绝对不杀 Ray) ----
echo "Cleaning up previous training processes..."
# 警告：千万不要执行 ray stop，也不要杀 raylet
pkill -9 -f agentevolver.main_ppo
pkill -9 -f vllm
# 注意：不要简单的 pkill -9 -f python，这可能会误杀 Ray 的 Python 组件
# 如果必须清理残留，建议精确匹配项目名

# ---- 2. 网络与代理配置 (Tencent Network) ----
export SETUPTOOLS_USE_DISTUTILS=local
export http_proxy=http://hk-mmhttpproxy.woa.com:11113
export https_proxy=http://hk-mmhttpproxy.woa.com:11113

# 获取本机 IP (用于 no_proxy)
HOST_IP=$(hostname -I | awk '{print $1}')
# 关键：加入 29.0.0.0/8 网段，防止 Ray 节点间通信走代理
export no_proxy="localhost,127.0.0.1,::1,.woa.com,$HOST_IP,29.0.0.0/8,10.0.0.0/8,172.16.0.0/12,$no_proxy"
export NO_PROXY="localhost,127.0.0.1,::1,.woa.com,$HOST_IP,29.0.0.0/8,10.0.0.0/8,172.16.0.0/12,$NO_PROXY"

# ---- 3. 启动环境服务 (AppWorld) ----
# 假设 appworld_multi.sh 就在当前目录
if [ -f "./env_service/launch_script/appworld_multi.sh" ]; then
    echo "Starting AppWorld Service..."
    bash ./env_service/launch_script/appworld_multi.sh > appworld_server.log 2>&1 &
    APPWORLD_PID=$!
    
    # 简单的健康检查等待 (等待 10 秒让服务启动)
    echo "Waiting 10s for AppWorld to initialize..."
    sleep 50
else
    echo "Error: appworld_multi.sh not found!"
    exit 1
fi

# 注册退出陷阱：脚本结束时关闭 AppWorld
trap "echo 'Stopping AppWorld...'; kill $APPWORLD_PID" EXIT

# ---- 4. 调用核心训练脚本 ----
echo "Starting Training Pipeline..."
bash ./Our_examples/run_api_driven_H20_multi.sh