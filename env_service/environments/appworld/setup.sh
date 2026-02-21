#!/bin/bash

# ==========================================
# AppWorld 环境配置脚本 (本地文件优先版)
# ==========================================

set -e
set -o pipefail

# 1. 路径定义
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
BEYONDAGENT_DIR="$(dirname "$(dirname "$ENV_SERVICE_DIR")")"
APPWORLD_ROOT="$SCRIPT_DIR"
WORKSPACE_DIR="$BEYONDAGENT_DIR"

# 关键：直接在当前目录查找 zip 文件
LOCAL_ZIP="$APPWORLD_ROOT/appworld_data.zip"
TEMP_DIR="/tmp/appworld_temp_$(date +%s)"

# 2. 环境变量
echo "🌐 配置环境变量..."
export no_proxy="localhost,127.0.0.1,::1,0.0.0.0,10.0.0.0/8,29.0.0.0/8,.woa.com,$no_proxy"
export NO_PROXY="localhost,127.0.0.1,::1,0.0.0.0,10.0.0.0/8,29.0.0.0/8,.woa.com,$NO_PROXY"
export NODE_ENV=production
export WORKSPACE_DIR="$WORKSPACE_DIR"
export APPWORLD_ROOT="$APPWORLD_ROOT"
export PYTHONPATH="$BEYONDAGENT_DIR:$PYTHONPATH"

# 3. Conda 环境检查
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

if ! conda info --envs | grep -w "appworld" &>/dev/null; then
    echo "🐍 创建 Conda 环境 appworld..."
    conda create -n appworld python=3.11.14 -y
else
    echo "✅ Conda 环境 appworld 已存在。"
fi

# 4. 初始化
echo "⚙️ 初始化 AppWorld..."
conda run -n appworld appworld install

# 5. 数据处理逻辑 (核心修改)
echo "📦 准备数据..."

# 判断当前目录下是否已经有人工上传的 zip 文件
if [ -f "$LOCAL_ZIP" ]; then
    echo "✅ 检测到本地已存在数据包: $LOCAL_ZIP"
    echo "   👉 跳过下载，直接解压..."
    
    mkdir -p "$TEMP_DIR"
    unzip -q -o "$LOCAL_ZIP" -d "$TEMP_DIR"
    
    echo "🚚 部署文件..."
    SUB_DIR=$(ls -d "$TEMP_DIR"/*/ 2>/dev/null | head -n 1)
    if [ -n "$SUB_DIR" ]; then
        cp -rf "$SUB_DIR"* "$APPWORLD_ROOT/"
    else
        cp -rf "$TEMP_DIR/"* "$APPWORLD_ROOT/"
    fi
    
    # 清理临时目录，但保留 zip 包以备下次重用
    rm -rf "$TEMP_DIR"
    echo "✅ 数据部署完成！"

else
    echo "❌ 未检测到本地数据包 ($LOCAL_ZIP)"
    echo "⚠️ 自动下载已多次失败，请执行以下手动操作："
    echo "-----------------------------------------------------"
    echo "1. 在本地电脑下载: https://dail-wlcb.oss-accelerate.aliyuncs.com/eric.czq/appworld_data.zip"
    echo "2. 上传到服务器目录: $APPWORLD_ROOT/"
    echo "3. 重新运行此脚本。"
    echo "-----------------------------------------------------"
    exit 1
fi

echo ""
echo "✅ ✅  所有设置已完成！"