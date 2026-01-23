#!/bin/bash
# ============================================================
# AppWorld 多机服务启动器
# ============================================================

export PYTHONUNBUFFERED=1

# 1. 路径自动计算 (回退两层到 test_agent 根目录)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"
export APPWORLD_ROOT="$PROJECT_ROOT/env_service/environments/appworld"
export PYTHONPATH="$PROJECT_ROOT:$APPWORLD_ROOT:$PYTHONPATH"

# 2. 激活环境
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate appworld

# 3. 启动
cd "$PROJECT_ROOT"
echo "📡 Starting AppWorld on 0.0.0.0:8080"
# 使用 exec 确保进程信号能被 launch_multi.sh 捕获
exec python -m env_service.env_service \
    --env appworld \
    --portal "0.0.0.0" \
    --port 8080 \
    --debug True