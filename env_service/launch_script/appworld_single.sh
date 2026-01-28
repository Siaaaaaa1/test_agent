#!/bin/bash

# ============================================================
# AppWorld 单机服务启动器 (被主脚本调用)
# ============================================================

# 1. 强制实时日志 (防止日志被吃)
export PYTHONUNBUFFERED=1

# 2. 网络配置 (绑定 0.0.0.0 以规避 localhost/127.0.0.1 差异)
export MASTER_ADDRESS="0.0.0.0" 
export PORT="8080"

# 获取本机 IP 用于 no_proxy
HOST_IP=$(hostname -I | awk '{print $1}')
export no_proxy="localhost,127.0.0.1,::1,0.0.0.0,29.209.112.175,.woa.com"
export NO_PROXY=$no_proxy

# 3. 路径计算 (自动定位项目根目录)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"   # .../env_service
PROJECT_ROOT="$(dirname "$ENV_SERVICE_DIR")" # .../test_agent
APPWORLD_DIR="$ENV_SERVICE_DIR/environments/appworld"

export APPWORLD_ROOT="$APPWORLD_DIR"
export PYTHONPATH="$PROJECT_ROOT:$APPWORLD_DIR:$PYTHONPATH"

# 4. 打印调试信息 (输出到主脚本的 server.log 中)
echo "========================================"
echo "🚀 AppWorld Service Launcher"
echo "📂 Project Root: $PROJECT_ROOT"
echo "📂 AppWorld Root: $APPWORLD_ROOT"
echo "========================================"

# 5. 自动激活 Conda (如果是被单独调用)
if [[ "$CONDA_DEFAULT_ENV" != "appworld" ]]; then
    echo "⚡ Activating 'appworld' conda environment..."
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate appworld
fi

# 确保 Ray 已安装
if ! python -c "import ray" 2>/dev/null; then
    echo "⚠️  Ray not found, installing..."
    pip install "ray[default]"
fi

# 6. 切换目录并启动
cd "$PROJECT_ROOT"

# 使用 exec 替换当前 Shell 进程，确保主脚本 kill 时能杀掉 Python
exec python -m env_service.env_service \
    --env appworld \
    --portal "$MASTER_ADDRESS" \
    --port "$PORT" \
    --debug True