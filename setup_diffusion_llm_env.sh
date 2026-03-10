#!/usr/bin/env bash
# =============================================================================
# diffusion-llm 环境配置脚本 (适配 CUDA 12.2，无法使用 cu128)
# 策略：优先安装核心包，次要包冲突则跳过
# =============================================================================

set -e
ENV_NAME="diffusion-llm"
PYTHON_VERSION="3.10"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "[1/6] 创建 conda 环境: $ENV_NAME (Python $PYTHON_VERSION)"
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

# 激活环境（脚本内用 source activate）
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "[2/6] 安装 PyTorch (CUDA 12.4，兼容 12.2 驱动)..."
pip install --upgrade pip
if ! pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124; then
  echo "  pip 安装 PyTorch 失败（多为网络/被墙），尝试 conda 安装..."
  conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
fi

echo "[3/6] 安装核心依赖 (Tier 1)..."
# tokenizers 需与 transformers 4.49.0 兼容：4.49.0 要求 tokenizers>=0.21,<0.22，故不装 0.22.1
pip install \
  transformers==4.49.0 \
  datasets==3.2.0 \
  accelerate==1.10.1 \
  "tokenizers>=0.21,<0.22" \
  safetensors==0.7.0 \
  huggingface_hub==0.35.3 \
  numpy \
  packaging \
  filelock \
  requests \
  tqdm \
  pyyaml \
  jinja2 \
  regex \
  sympy \
  networkx \
  fsspec

echo "[4/6] 安装 Tier 2 (flash-attn / deepspeed / peft 等)..."
# flash-attn: 使用与 cu124 兼容的版本，不锁 2.7.4.post1（该版本面向 cu128）
pip install flash-attn --no-build-isolation || pip install flash-attn || echo "  [跳过] flash_attn 安装失败，可后续从源码或预编译 wheel 安装"

pip install deepspeed || echo "  [跳过] deepspeed"
pip install peft==0.14.0 || pip install peft || echo "  [跳过] peft"
pip install lm_eval==0.4.9 || pip install lm_eval || echo "  [跳过] lm_eval"
pip install einops==0.8.1
pip install fire==0.7.1
pip install wandb || true
pip install tensorboard || true

echo "[5/6] 安装 Tier 3 (其余主要包，版本放宽或冲突则跳过)..."
pip install \
  aiohttp \
  appdirs \
  attrs \
  beautifulsoup4 \
  bert_score \
  boto3 \
  cohere \
  dill \
  easydict \
  fastapi \
  filelock \
  gradio \
  gradio_client \
  hf_transfer \
  html2text \
  httpx \
  hydra-core \
  immutabledict \
  jieba \
  jsonlines \
  jsonschema \
  langcodes \
  langdetect \
  more_itertools \
  nltk \
  omegaconf \
  optimum \
  pandas \
  Pillow \
  psutil \
  pycountry \
  pytest \
  python-dotenv \
  radgraph \
  regex \
  rich \
  rouge_score \
  sacrebleu \
  scikit-learn \
  scipy \
  sentence-transformers \
  setuptools \
  sqlitedict \
  tabulate \
  tenacity \
  tensordict \
  tiktoken \
  tqdm \
  uvicorn \
  wget \
  zstandard \
  xxhash \
  || true

# 以下包易与 PyTorch/CUDA 或其它包冲突，单独安装且失败不中断
for pkg in \
  "triton" \
  "lightning" \
  "mamba_ssm" \
  "sglang" \
  "nemo" \
  "auto_gptq" \
  "deepspeed" \
  "lmflow" \
  "transformer_lens" \
  "sae_lens" \
  "veomni" \
  "verl" \
  "hidet" \
  "fastchat" \
  "promptsource" \
  "kstar_planner" \
  "math_verify" \
  "stop_sequencer" \
  "sandbox_fusion" \
  "liger_kernel" \
  "django_cache_utils" \
  "e2b" \
  "mistralai" \
  "zeno_client" \
  "cirron" \
  "codetiming" \
  "pddl" \
  "tarski" \
  "tilus" \
  "word2number" \
  "wonderwords" \
  "tree_sitter" \
  "tree_sitter_python" \
  "pymorphy2" \
  "fugashi" \
  "neologdn" \
  "emoji" \
  "fuzzywuzzy" \
  "pytablewriter" \
  "pympler" \
  "torchdata" \
  "tempdir" \
  "pebble" \
  "pqdm" \
  "astor" \
  "multipledispatch" \
  "lark" \
  "matplotlib" \
  "black" \
  "coverage" \
  "termcolor" \
  "tqdm_multiprocess" \
  "botocore" \
  "sparsify" \
  "ray" \
  ; do
  pip install "$pkg" 2>/dev/null || echo "  [跳过] $pkg"
done

echo "[6/6] 从 requirements.txt 查漏补缺 (忽略版本，仅补装缺失)..."
# 只安装当前环境里没有的包名，避免覆盖已有版本
while IFS= read -r line || [[ -n "$line" ]]; do
  line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  pkg_name=$(echo "$line" | sed 's/[<=>].*//')
  pkg_name=$(echo "$pkg_name" | sed 's/\[.*//')
  # 跳过已在前面专门处理的（torch 用 cu124；tokenizers 与 transformers 4.49 兼容用 0.21.x）
  [[ "$pkg_name" == "torch" || "$pkg_name" == "tokenizers" ]] && continue
  pip show "$pkg_name" &>/dev/null || pip install "$pkg_name" 2>/dev/null || true
done < "$REPO_ROOT/requirements.txt"

echo ""
echo "=============================================="
echo "环境 '$ENV_NAME' 配置完成。"
echo "请执行: conda activate $ENV_NAME"
echo "若缺少某包可手动: pip install <包名>"
echo "若 flash_attn 未装上，可到 https://flashattn.dev/install 选 cu124 获取预编译 wheel"
echo "=============================================="
