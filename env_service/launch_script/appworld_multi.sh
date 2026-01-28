#!/bin/bash

# ==========================================
# AppWorld 分布式服务启动脚本 (最终路径修正版)
# ==========================================

# ---- 1. 网络配置 ----
KNOWN_MASTER_IP="29.209.112.175"
HOST_IP=$(hostname -I | awk '{print $1}')
BIND_ADDRESS="0.0.0.0"

# 代理设置
export no_proxy="localhost,127.0.0.1,::1,0.0.0.0,$BIND_ADDRESS,$KNOWN_MASTER_IP,$HOST_IP,29.0.0.0/8,10.0.0.0/8,.woa.com,$no_proxy"
export NO_PROXY="localhost,127.0.0.1,::1,0.0.0.0,$BIND_ADDRESS,$KNOWN_MASTER_IP,$HOST_IP,29.0.0.0/8,10.0.0.0/8,.woa.com,$NO_PROXY"

# ---- 2. 路径配置 (根据你提供的路径修正) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. 项目根目录 (test_agent)
# SCRIPT_DIR 是 .../test_agent/env_service/launch_script
# 回退两层得到 test_agent
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"

# 2. [核心修复] 定位 APPWORLD_ROOT
# 你的路径是: /mnt/cephfs/haowengao/test_agent/env_service/environments/appworld
# 所以它在 PROJECT_ROOT 下的 env_service/environments/appworld
export APPWORLD_ROOT="$PROJECT_ROOT/env_service/environments/appworld"

# 双重检查：如果动态计算失败，直接使用你提供的硬编码绝对路径
if [ ! -d "$APPWORLD_ROOT" ]; then
    echo "Warning: Calculated path not found, using hardcoded path..."
    export APPWORLD_ROOT="/mnt/cephfs/haowengao/test_agent/env_service/environments/appworld"
fi

# 检查路径是否存在
if [ ! -d "$APPWORLD_ROOT" ]; then
    echo "CRITICAL ERROR: APPWORLD_ROOT directory does not exist!"
    echo "Path: $APPWORLD_ROOT"
    exit 1
fi

export RAY_ENV_NAME=appworld
# PYTHONPATH 必须包含 Project Root，这样才能运行 python -m env_service...
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "Config:"
echo "  - Project Root:  $PROJECT_ROOT"
echo "  - AppWorld Root: $APPWORLD_ROOT"
echo "  - Bind Address:  $BIND_ADDRESS"

# ---- 3. 环境激活 ----
echo "Activating Conda Environment: appworld..."
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi
conda activate appworld

if [ "$CONDA_DEFAULT_ENV" != "appworld" ]; then
    echo "Error: Failed to activate conda environment 'appworld'!"
    exit 1
fi

# ---- 4. 清理旧进程 ----
pkill -f "env_service.env_service" || true
sleep 1

# ---- 5. 启动服务 ----
# 切换到项目根目录运行
cd "$PROJECT_ROOT"
echo "Starting AppWorld Server on $BIND_ADDRESS:8080..."

exec python -m env_service.env_service \
    --env appworld \
    --portal "$BIND_ADDRESS" \
    --port 8080