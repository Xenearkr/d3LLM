#!/usr/bin/env bash
# d3LLM Gradio 可视化环境配置
# 用法: bash gradio/setup_env.sh  （在 d3LLM 项目根目录执行）

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

ENV_NAME="d3llm-gradio"

echo "===== 创建 conda 环境 ${ENV_NAME} (Python 3.10) ====="
if conda env list | grep -q "^${ENV_NAME} "; then
  echo "环境 ${ENV_NAME} 已存在，跳过创建"
else
  if conda env list | grep -q "^d3llm "; then
    echo "从已有 d3llm 环境克隆（更快）..."
    conda create -n "${ENV_NAME}" --clone d3llm -y
  else
    conda create -n "${ENV_NAME}" python=3.10 -y
  fi
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "===== 安装 PyTorch (CUDA 12.8) ====="
if python -c "import torch" 2>/dev/null; then
  echo "PyTorch 已存在，跳过安装"
else
  pip install --upgrade pip
  pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
fi

echo "===== 安装核心依赖 ====="
pip install transformers==4.49.0 "tokenizers>=0.21,<0.22" safetensors einops

echo "===== 安装 Gradio 依赖 ====="
pip install -r gradio/requirements.txt

echo "===== 验证 ====="
python -c "
import torch, transformers, gradio
print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('transformers:', transformers.__version__)
print('gradio:', gradio.__version__)
print('环境 ${ENV_NAME} 配置完成')
"

echo ""
echo "激活环境:  conda activate ${ENV_NAME}"
echo "启动界面:  cd $ROOT_DIR && python gradio/app.py"
echo "可选镜像:  HF_ENDPOINT=https://hf-mirror.com python gradio/app.py"
