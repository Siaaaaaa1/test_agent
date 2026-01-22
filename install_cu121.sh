#!/usr/bin/env bash
set -e
export PYTHONNOUSERSITE=1
echo "======================================================"
echo "   Installing AgentEvolver Environment (Local Verl)"
echo "   Config: PyTorch 2.5.1 | CUDA 12.1 | vLLM 0.8.5"
echo "======================================================"
echo

# ---- Step 0. Check Local Verl Path ----
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
ENV_NAME="AgentEvolver121"
PYTHON_VERSION="3.11.14" 

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
echo "🚀 Installing CUDA toolkit 12.1 and Build Tools..."
conda install -y -c nvidia cuda-toolkit=12.1
pip install --upgrade pip wheel setuptools ninja packaging

# ---- Step 6. Install PyTorch 2.5.1 (cu121) ----
echo
echo "🔥 Installing PyTorch 2.5.1 (cu121)..."
# 注意：这里移除了 PyTorch 2.6 的安装命令，仅保留 2.5.1
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# ---- Step 7. Install Flash Attention (Wheel for Torch 2.5) ----
echo
echo "⚡ Installing Flash Attention (Wheel for Torch 2.5)..."
# 使用适配 Torch 2.5 和 Python 3.11 的 Wheel 包
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu121torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# ---- Step 8. Install Core LLM Inference Libraries ----
echo
echo "🧠 Installing vLLM 0.8.5 and FlashInfer..."
# 安装 vLLM 0.8.5 (适配 PyTorch 2.5)
# pip install vllm==0.8.5
pip install https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu121-cp38-abi3-manylinux1_x86_64.whl
pip install flashinfer-python==0.3.1

# ---- Step 9. Install General Dependencies ----
    echo
    echo "📥 Installing remaining packages from requirements_cu121.txt..."
    pip install -r requirements_cu121.txt

# ---- Step 10. Install Verl (Local Source) ----
echo

echo "🛠️  Installing Verl from local path: ./external/verl ..."
# --no-build-isolation 确保使用当前环境已安装的 torch 和 flash-attn
pip install -r ./external/verl/requirements_cu121.txt
pip install -e ./external/verl --no-build-isolation --no-deps

# ---- Step 11. Finish ----
echo
echo "✅ Installation complete!"
echo "To start using the environment, run:"
echo "  conda activate $ENV_NAME"
echo