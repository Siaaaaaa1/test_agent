#!/bin/bash

# ---- 0. 修改：强制指定为 Localhost ----
# [原代码] export MASTER_ADDRESS=$(ip route get 1.1.1.1 | grep -oP 'src \K\S+')
# [新代码] 强制绑定本地，配合之前的训练脚本
export MASTER_ADDRESS="127.0.0.1"

# ---- 新增：防止代理拦截本地服务 ----
# 即使是服务端，加上这个也能防止 Python 内部请求走错代理
export no_proxy="localhost,127.0.0.1,::1,.woa.com,$no_proxy"
export NO_PROXY="localhost,127.0.0.1,::1,.woa.com,$NO_PROXY"

# 可以修改为自己的appworld数据路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
BEYONDAGENT_DIR="$(dirname "$(dirname "$ENV_SERVICE_DIR")")"
APPWORLD_ROOT="${APPWORLD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../environments/appworld" && pwd)}"
export APPWORLD_ROOT
echo "APPWORLD_ROOT: $APPWORLD_ROOT"

# Ray 环境名配置
export RAY_ENV_NAME=appworld

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 导航到项目根目录 (env_service)
PROJECT_ROOT="$SCRIPT_DIR/../../"
cd "$PROJECT_ROOT"

# 设置 PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 打印调试信息
echo "Current working directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"
echo "Server will listen on: $MASTER_ADDRESS:8080"

# 运行 Python 命令
# --portal 参数现在会被传入 "127.0.0.1"
exec python -m env_service.env_service \
    --env appworld \
    --portal "$MASTER_ADDRESS" \
    --port 8080