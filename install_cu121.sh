#!/usr/bin/env bash
set -e  # 遇到错误立即退出

# ================= 配置部分 =================
ENV_NAME="agentevolver"
PYTHON_VERSION="3.11.14"
# ===========================================

export PYTHONNOUSERSITE=1
echo "======================================================"
echo "   Installing AgentEvolver Environment (Resumable)"
echo "   Config: PyTorch 2.5.1 | CUDA 12.1 | vLLM 0.8.5"
echo "======================================================"
echo

# ---- Step 0. Check Local Verl Path ----
if [[ ! -d "./external/verl" ]]; then
    echo "❌ Error: Directory './external/verl' not found!"
    exit 1
fi

# 初始化 Conda (确保脚本中可以使用 conda 命令)
# 尝试找到 conda.sh 的位置
CONDA_BASE=$(conda info --base)
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "❌ Error: Could not verify conda installation."
    exit 1
fi

# ---- Step 1-4. Create or Activate Environment ----
if conda info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "✅ Environment '$ENV_NAME' already exists. Resuming installation..."
    conda activate "$ENV_NAME"
else
    echo "📦 Creating environment '$ENV_NAME'..."
    conda create -y -n "$ENV_NAME" python=$PYTHON_VERSION
    conda activate "$ENV_NAME"
fi

# 确保路径正确
export PATH=$CONDA_PREFIX/bin:$PATH

# ---- 设置状态标记目录 ----
# 我们在环境目录下创建一个隐藏文件夹来记录进度
MARKER_DIR="$CONDA_PREFIX/.install_markers"
mkdir -p "$MARKER_DIR"

# 定义检测函数
function check_step() {
    local step_name=$1
    if [[ -f "$MARKER_DIR/$step_name" ]]; then
        return 0 # 已完成
    else
        return 1 # 未完成
    fi
}

function mark_step_done() {
    local step_name=$1
    touch "$MARKER_DIR/$step_name"
    echo "🎉 Step '$step_name' marked as complete."
}

# ---- Step 5. Install CUDA & Build Tools ----
echo
echo "🚀 [Step 5] Checking CUDA toolkit 12.1..."

if check_step "step5_cuda_installed"; then
    echo "   -> Skipped (Already installed)"
else
    conda install -y -c nvidia cuda-toolkit=12.1
    mark_step_done "step5_cuda_installed"
fi

# ⚠️ 注意：环境变量必须每次运行都设置，不能跳过
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
echo "   -> CUDA_HOME set to $CUDA_HOME"

if ! check_step "step5_pip_upgraded"; then
    pip install --upgrade pip wheel setuptools ninja packaging
    mark_step_done "step5_pip_upgraded"
fi

# ---- Step 6. Install PyTorch 2.5.1 ----
echo
echo "🔥 [Step 6] Checking PyTorch 2.5.1..."

if check_step "step6_torch_installed"; then
    echo "   -> Skipped (Already installed)"
else
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    mark_step_done "step6_torch_installed"
fi

# ---- Step 7. Install Flash Attention ----
echo
echo "⚡ [Step 7] Checking Flash Attention..."

if check_step "step7_flash_attn_installed"; then
    echo "   -> Skipped (Already installed)"
else
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
    mark_step_done "step7_flash_attn_installed"
fi

# ---- Step 8. Install General Dependencies ----
echo
echo "📥 [Step 8] Checking requirements_cu121.txt..."

if check_step "step8_requirements_installed"; then
    echo "   -> Skipped (Already installed)"
else
    if [[ -f requirements_cu121.txt ]]; then
        pip install -r requirements_cu121.txt
        mark_step_done "step8_requirements_installed"
    else
        echo "⚠️  Warning: requirements_cu121.txt not found, skipping."
    fi
fi

# ---- Step 9. Compile vLLM 0.8.5 (Golden Fix: Label Channel 12.1) ----
echo
echo "🏗️  [Step 9] Checking vLLM 0.8.5 compilation..."

if check_step "step9_vllm_installed"; then
    echo "   -> Skipped (Already installed)"
else
    echo "   Beginning vLLM build process..."

    # 1. 【终极修复】使用 Label Channel 锁定 12.1
    # 我们不再使用 'cuda-toolkit=12.1' 这种软约束
    # 而是强制指定频道 nvidia/label/cuda-12.1.1，这里面根本没有 12.8 的包
    echo "   🔧 Installing STRICT CUDA 12.1 toolchain..."
    
    # 必须先卸载可能的冲突包，防止 conda 解决依赖失败
    conda uninstall -y cuda-toolkit cuda-nvtx cuda-tools || true

    # 安装核心组件 (去掉了不存在的 cuda-headers)
    conda install -y \
        -c "nvidia/label/cuda-12.1.1" \
        -c conda-forge \
        cuda-toolkit \
        cuda-nvtx \
        cuda-tools \
        gxx_linux-64=12 gcc_linux-64=12 sysroot_linux-64

    # 2. 【关键】修复库路径软链接 (解决 NVTX 找不到的问题)
    echo "   🔧 Fix: Creating library symlinks..."
    
    # 欺骗 CMake 让他找到 lib64
    if [[ ! -L "$CONDA_PREFIX/lib64" ]]; then
        ln -s "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"
    fi

    # 修复缺失的 libnvToolsExt.so
    # 很多时候 Conda 只给 libnvToolsExt.so.1，但 CMake 只找 .so
    find "$CONDA_PREFIX/lib" -name "libnvToolsExt.so.*" | while read -r file; do
        base_name=$(basename "$file")
        # 目标链接名：libnvToolsExt.so
        link_name="libnvToolsExt.so"
        if [[ ! -f "$CONDA_PREFIX/lib/$link_name" ]]; then
            ln -sf "$base_name" "$CONDA_PREFIX/lib/$link_name"
            echo "     -> Repaired symlink: $link_name -> $base_name"
        fi
    done

    # 3. 暴露 targets 目录下的头文件 (防止 cuda.h 找不到)
    # CUDA 12 的 Conda 包经常把头文件藏在 targets/x86_64-linux/include 里
    TARGET_DIR="$CONDA_PREFIX/targets/x86_64-linux"
    if [[ -d "$TARGET_DIR" ]]; then
        echo "   🔧 Fix: Exposing internal CUDA headers..."
        cp -rn "$TARGET_DIR/include/"* "$CONDA_PREFIX/include/" || true
        cp -rn "$TARGET_DIR/lib/"* "$CONDA_PREFIX/lib/" || true
    fi

    # 4. 准备源码
    if [[ ! -d "vllm_build" ]]; then
        git clone https://github.com/vllm-project/vllm.git vllm_build
    fi
    
    pushd vllm_build > /dev/null
    
    # 5. 清理旧缓存 (非常重要！否则 CMake 会记住 12.8 的路径)
    echo "   🧹 Cleaning build artifacts..."
    rm -rf build vllm/*.so .deps
    pip cache purge
    
    # 6. 切换 vLLM 版本
    git fetch --all
    git checkout v0.8.5
    python use_existing_torch.py
    pip install -r requirements/build.txt
    
    # 7. 设置编译环境变量
    echo "   🔧 Setting build environment..."
    
    # 编译器 (GCC 12)
    export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
    export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
    export NVCC_CCBIN=$CXX
    
    # CUDA 路径
    export CUDA_HOME=$CONDA_PREFIX
    export CUDA_ROOT=$CONDA_PREFIX
    export CUDAToolkit_ROOT=$CONDA_PREFIX
    
    # 路径注入
    export CMAKE_PREFIX_PATH=$CONDA_PREFIX:$CMAKE_PREFIX_PATH
    export LDFLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib $LDFLAGS"
    export CFLAGS="-I$CONDA_PREFIX/include $CFLAGS"
    export CXXFLAGS="-I$CONDA_PREFIX/include $CXXFLAGS"

    echo "   Compiling with MAX_JOBS=8..."
    export MAX_JOBS=8
    
    # 8. 开始编译
    # 显式指定 nvcc 路径，防止使用系统默认的
    export CMAKE_ARGS="-DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX \
                       -DCUDAToolkit_ROOT=$CONDA_PREFIX \
                       -DCUDA_HOME=$CONDA_PREFIX \
                       -DCMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc \
                       -DVLLM_TARGET_DEVICE=cuda"

    pip install --no-build-isolation -e .
    
    popd > /dev/null
    
    mark_step_done "step9_vllm_installed"
fi

# ---- Step 10. Install Verl ----
echo
echo "🛠️  [Step 10] Checking Verl installation..."

if check_step "step10_verl_installed"; then
    echo "   -> Skipped (Already installed)"
else
    pip install -e ./external/verl --no-build-isolation --no-deps
    mark_step_done "step10_verl_installed"
fi

echo
echo "✅ All Steps Completed! Environment '$ENV_NAME' is ready."
echo "To activate: conda activate $ENV_NAME"