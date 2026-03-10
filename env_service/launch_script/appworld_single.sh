#!/bin/bash

# 开启严格模式：遇到执行错误即退出 (除了条件判断里)
set -e

# ============================================================
# AppWorld 单机服务启动器 (Tencent WOA 修复版 - 优化增强)
# ============================================================

# 1. 强制实时日志 (防止日志被吃)
export PYTHONUNBUFFERED=1

# 2. 检查必须的环境变量 (并提供缺省值)
MASTER_ADDRESS=${MASTER_ADDRESS:-"127.0.0.1"} # 默认本机
PORT=${PORT:-"8080"}                          # 默认8080端口

# 3. 路径计算 (自动定位项目根目录与工作区)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"   # .../env_service
PROJECT_ROOT="$(dirname "$ENV_SERVICE_DIR")" # 例如: /mnt/workspace/username/AgentEvolver
WORKSPACE_DIR="$(dirname "$PROJECT_ROOT")"   # 动态获取工作区目录: /mnt/workspace/username

APPWORLD_DIR="$ENV_SERVICE_DIR/environments/appworld"

export APPWORLD_ROOT="$APPWORLD_DIR"
export PYTHONPATH="$PROJECT_ROOT:$APPWORLD_DIR:$PYTHONPATH"

# 4. 打印调试信息
echo "========================================"
echo "🚀 AppWorld Service Launcher"
echo "📂 Project Root: $PROJECT_ROOT"
echo "📂 Workspace: $WORKSPACE_DIR"
echo "📂 AppWorld Root: $APPWORLD_ROOT"
echo "🌐 Portal: $MASTER_ADDRESS:$PORT"
echo "========================================"

# 5. 自动激活 Conda
if [[ "$CONDA_DEFAULT_ENV" != "appworld" ]]; then
    echo "⚡ Activating 'appworld' conda environment..."
    
    # 动态寻找确切的 conda 路径 (优先检查当前项目所在的工作区目录)
    if [ -f "$WORKSPACE_DIR/miniconda3/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$WORKSPACE_DIR/miniconda3"
    elif [ -f "$WORKSPACE_DIR/anaconda3/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$WORKSPACE_DIR/anaconda3"
    elif command -v conda &> /dev/null; then
        CONDA_BASE=$(conda info --base)
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$HOME/anaconda3"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$HOME/miniconda3"
    else
        echo "❌ Error: Cannot find conda installation."
        exit 1
    fi
    
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    
    # 尝试激活，如果失败则报错退出 (通过 || exit 1 拦截)
    conda activate appworld || {
        echo "❌ Error: Failed to activate conda environment 'appworld'. Please check if it exists."
        exit 1
    }
fi

# 6. 确保 Ray 已安装
if ! python -c "import ray" 2>/dev/null; then
    echo "⚠️  Ray not found, installing..."
    pip install "ray[default]"
fi

# 7. 切换目录并启动
cd "$PROJECT_ROOT"

echo "🌟 Starting service..."
# 使用 exec 替换当前 Shell 进程，并安全传入变量
exec python -m env_service.env_service \
    --env appworld \
    --portal "$MASTER_ADDRESS" \
    --port "$PORT" \
    --debug True