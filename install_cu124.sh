#!/usr/bin/env bash
set -e
export PYTHONNOUSERSITE=1
echo "======================================================"
echo "   Installing AgentEvolver Environment (Local Verl)"
echo "======================================================"
echo

# ---- Step 0. Check Local Verl Path ----
# 先检查本地路径是否存在，避免跑了一半报错
if [[ ! -d "./external/verl" ]]; then
    echo "❌ Error: Directory './external/verl' not found!"
    echo "   Please make sure you are running this script from the project root"
    echo "   and that 'external/verl' exists."
    exit 1
fi

# ---- Step 1. Check Conda installation ----
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed or not found in PATH."
    exit 1
fi

# ---- Step 2. Setup Environment Name ----
ENV_NAME="AgentEvolver"
PYTHON_VERSION="3.11.14" # Must match the FlashAttn wheel (cp311)

if conda info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "⚠️  Environment '$ENV_NAME' already exists."
    echo "   Please remove it first if you want a fresh install: conda env remove -n $ENV_NAME"
    exit 1
fi

# ---- Step 3. Create new environment ----
echo
echo "📦 Creating environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -y -n "$ENV_NAME" python=$PYTHON_VERSION
export PATH=$CONDA_PREFIX/bin:$PATH

# ---- Step 4. Activate environment ----
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ---- Step 5. Install Basic Build Tools & CUDA ----
echo
echo "🚀 Installing CUDA toolkit 12.4 and Build Tools..."
conda install -y -c nvidia cuda-toolkit=12.4
# ninja 和 packaging 是编译 verl 及其依赖所必需的
pip install --upgrade pip wheel setuptools ninja packaging

# ---- Step 6. Install PyTorch (Specific Version) ----
echo
echo "🔥 Installing PyTorch 2.6.0 (cu124)..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com

# ---- Step 7. Install Flash Attention (Pre-compiled Wheel) ----
echo
echo "⚡ Installing Flash Attention 2.8.3 (Wheel)..."
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# ---- Step 8. Install Core LLM Inference Libraries ----
echo
echo "🧠 Installing vLLM 0.8.5 and FlashInfer..."
pip install vllm==0.8.5
pip install flashinfer-python==0.3.1

# ---- Step 9. Install General Dependencies ----
if [[ ! -f requirements.txt ]]; then
    echo "⚠️  No requirements.txt found. Skipping..."
else
    echo
    echo "📥 Installing remaining packages from requirements.txt..."
    pip install -r requirements.txt
fi

# ---- Step 10. Install Verl (Local Source) ----
echo
echo "🛠️  Installing Verl from local path: ./external/verl ..."
# 使用 -e 模式安装，方便开发调试
# --no-build-isolation 至关重要，确保 verl 编译时能找到刚才安装好的 flash-attn 和 torch
pip install -e ./external/verl --no-build-isolation

# ---- Step 11. Finish ----
echo
echo "✅ Installation complete!"
echo "To start using the environment, run:"
echo "  conda activate $ENV_NAME"
echo