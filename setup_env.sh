#!/usr/bin/env bash
# d3LLM 环境配置脚本：创建 conda 环境 d3llm，安装 CUDA 工具链、PyTorch、FlashAttention 及依赖
# 使用方式: bash setup_env.sh  （在 d3LLM 项目根目录执行）

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "===== 1. 创建 conda 环境 d3llm (Python 3.10) ====="
conda create -n d3llm python=3.10 -y

echo "===== 2. 激活环境并安装 CUDA 工具链（提供 nvcc，用于编译 FlashAttention）====="
# 在脚本中正确激活 conda 环境
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate d3llm
# 使用 conda-forge 的 cuda-toolkit 和 cuda-nvcc，与驱动 CUDA 12.2 兼容
conda install -c conda-forge cuda-toolkit cuda-nvcc -y

# 确保 nvcc 在 PATH 中
export PATH="$CONDA_PREFIX/bin:$PATH"
export CUDA_HOME="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

echo "===== 3. 安装 PyTorch 2.7.1 (CUDA 12.8) ====="
pip install --upgrade pip
pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "===== 4. 安装 FlashAttention 编译依赖 ====="
pip install ninja packaging wheel setuptools psutil -U

echo "===== 5. 安装 FlashAttention 2.7.4.post1 ====="
# 优先使用官方预编译 wheel（cu12+torch2.7+cp310），避免源码编译的跨设备链接问题
FLASH_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
if pip install "$FLASH_WHEEL" 2>/dev/null; then
  echo "FlashAttention: 已通过预编译 wheel 安装"
else
  echo "FlashAttention: 预编译 wheel 不可用，尝试从源码编译 (RTX A6000 = sm_86)..."
  export FLASH_ATTN_CUDA_ARCHS=86
  pip install flash-attn==2.7.4.post1 --no-build-isolation
fi

echo "===== 6. 安装核心依赖（transformers/datasets/lm_eval/accelerate/deepspeed）====="
pip install transformers==4.49.0 datasets==3.2.0 lm_eval==0.4.9 accelerate==1.10.1 deepspeed==0.16.2

echo "===== 6b. 可选：安装 requirements.txt 中其余依赖（存在版本冲突时可跳过）====="
echo "  pip install -r requirements.txt  # 若失败，可仅使用上述核心依赖运行 chat 与训练"

echo "===== 7. 验证安装 ====="
python -c "
import torch
import transformers
import flash_attn
print('PyTorch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())
print('transformers:', transformers.__version__)
print('flash_attn:', flash_attn.__version__)
print('Environment d3llm configured successfully.')
"

echo ""
echo "环境配置完成。使用以下命令激活环境："
echo "  conda activate d3llm"
echo "运行 Demo："
echo "  python chat/chat_d3llm_dream.py"
echo "  python chat/chat_d3llm_llada.py"

# 设置 Hugging Face 镜像，解决 Connection reset by peer 问题
export HF_ENDPOINT="https://hf-mirror.com"