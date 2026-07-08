#!/usr/bin/env bash
# 启动 d3LLM Dream-Coder SGLang 服务（OpenAI 兼容 API）
set -eo pipefail

_script_path="${BASH_SOURCE[0]}"
[[ "${_script_path}" != /* ]] && _script_path="$(pwd)/${_script_path}"
SCRIPT_DIR="$(cd "$(dirname "${_script_path}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
unset _script_path

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/finetune_d3LLM_DREAM_Coder}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-d3llm-dream-coder}"
BIND_HOST="${BIND_HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
API_KEY="${API_KEY:-sk-d3llm-local}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/models/sglang_d3llm_coder.log}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"

if [[ -n "${PYTHON:-}" ]]; then
  :
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${CONDA_PREFIX}/bin/python"
else
  PYTHON="$(command -v python3)"
fi

ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-1}"

export CUDA_VISIBLE_DEVICES
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
if [[ -d "${CONDA_PREFIX:-}/bin" ]]; then
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
fi
if [[ -d /usr/local/cuda/lib64/stubs ]]; then
  export LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH:-}"
fi

echo "[sglang] model=${MODEL_PATH}"
echo "[sglang] served_name=${SERVED_MODEL_NAME} bind_host=${BIND_HOST} port=${PORT}"
echo "[sglang] context_length=${CONTEXT_LENGTH} mem_fraction_static=${MEM_FRACTION_STATIC}"
echo "[sglang] attention_backend=${ATTENTION_BACKEND} disable_cuda_graph=${DISABLE_CUDA_GRAPH}"
echo "[sglang] python=${PYTHON} log=${LOG_FILE}"

SGLANG_ARGS=(
  --model-path "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${BIND_HOST}"
  --port "${PORT}"
  --api-key "${API_KEY}"
  --trust-remote-code
  --attention-backend "${ATTENTION_BACKEND}"
  --dllm-algorithm FullAttnMultiBlock
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --context-length "${CONTEXT_LENGTH}"
)

if [[ "${DISABLE_CUDA_GRAPH}" == "1" ]]; then
  SGLANG_ARGS+=(--disable-cuda-graph)
else
  SGLANG_ARGS+=(--cuda-graph-max-bs 8)
fi

exec "${PYTHON}" -m sglang.launch_server "${SGLANG_ARGS[@]}" \
  2>&1 | tee -a "${LOG_FILE}"
