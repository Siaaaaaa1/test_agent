#!/bin/bash

# ============================================================
# AgentEvolver 训练流水线 (自动管理 AppWorld 服务版 - Conda隔离启动)
# ============================================================

# ---- 1. 清理旧进程 (确保 8080 端口不被占用) ----
echo "Cleaning up previous processes..."
# 杀掉之前的训练任务
pkill -9 -f agentevolver.main_ppo
# 杀掉之前的 vllm (如果有)
pkill -9 -f vllm
# ⭐ 新增：确保杀掉旧的 AppWorld 服务，防止端口冲突
pkill -f "env_service.env_service"
# 强力确保 8080 空闲
fuser -k -9 8080/tcp >/dev/null 2>&1

sleep 2

# ---- 2. 网络与代理配置 (核心修复) ----
# 必须显式设置 no_proxy，否则 AppWorld 会拒绝 Ray 的连接 (403 Forbidden)
export SETUPTOOLS_USE_DISTUTILS=local
export http_proxy=http://hk-mmhttpproxy.woa.com:11113
export https_proxy=http://hk-mmhttpproxy.woa.com:11113
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=bond1
export TP_SOCKET_IFNAME=bond1
# 获取本机 IP
HOST_IP=$(hostname -I | awk '{print $1}')

# ⭐ 关键配置：必须包含 AppWorld IP (29.209.112.175) 和本机 IP
# 这样 Python 代码发请求时才不会走公司代理，从而避免 403 错误
export no_proxy="localhost,127.0.0.1,::1,0.0.0.0,29.209.112.175,$HOST_IP,29.0.0.0/8,10.0.0.0/8,172.16.0.0/12,.woa.com"
export NO_PROXY=$no_proxy

echo "Proxy Configured correctly."

# ---- 3. 启动环境服务 (AppWorld) ----
# 恢复了原来的逻辑：脚本负责启动服务
if [ -f "./env_service/launch_script/appworld_multi_v2.sh" ]; then
    echo "Starting AppWorld Service in conda env 'appworld'..."
    
    # ⭐ 修改点：在子 Shell 中激活 appworld 环境并启动，避免污染主脚本环境
    (
        # 尝试自动获取 conda base 路径，如果失败则回退到默认路径
        # 注意：如果你使用的是 miniconda，且不在默认路径，可能需要手动指定，例如：
        # CONDA_BASE="/mnt/cephfs/haowengao/miniconda3"
        CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")
        
        if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
        else
            echo "Warning: Conda profile not found at $CONDA_BASE/etc/profile.d/conda.sh, trying to rely on PATH..."
        fi
        
        echo "Activating appworld environment..."
        conda activate appworld
        
        # 后台启动服务，日志输出到 appworld_server.log
        bash ./env_service/launch_script/appworld_multi_v2.sh > appworld_server.log 2>&1
    ) &
    
    # 获取整个后台子 Shell 的 PID
    APPWORLD_PID=$!
    
    # 等待服务完全启动 (建议稍微久一点，防止 Uvicorn 还没这就绪训练就开始了)
    echo "Waiting 30s for AppWorld to initialize..."
    sleep 30
else
    echo "Error: appworld_multi_v2.sh not found!"
    exit 1
fi

# 注册退出陷阱：无论训练成功还是失败，脚本退出时都会自动关闭 AppWorld
trap "echo 'Stopping AppWorld (PID $APPWORLD_PID)...'; kill $APPWORLD_PID" EXIT

# ---- 4. 调用核心训练脚本 ----
echo "Starting Training Pipeline..."
# 训练脚本会自动继承上面的 export no_proxy 设置，但在原来的 Conda 环境中运行
bash ./Our_examples/run_api_driven_H20_multi.sh