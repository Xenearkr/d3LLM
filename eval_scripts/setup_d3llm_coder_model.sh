#!/usr/bin/env bash
# 下载 XenonGas/finetune_d3LLM_DREAM_Coder 到 models/finetune_d3LLM_DREAM_Coder
set -eo pipefail

_script_path="${BASH_SOURCE[0]}"
[[ "${_script_path}" != /* ]] && _script_path="$(pwd)/${_script_path}"
SCRIPT_DIR="$(cd "$(dirname "${_script_path}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
unset _script_path

MODEL_ID="${MODEL_ID:-XenonGas/finetune_d3LLM_DREAM_Coder}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/models/finetune_d3LLM_DREAM_Coder}"

PYTHON="${PYTHON:-python3}"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${CONDA_PREFIX}/bin/python"
fi

mkdir -p "$(dirname "${MODEL_DIR}")"

if [[ -L "${MODEL_DIR}" || -d "${MODEL_DIR}" ]] && \
   compgen -G "${MODEL_DIR}/model-*-of-*.safetensors" >/dev/null; then
  echo "[model] 已存在: ${MODEL_DIR}"
  ls "${MODEL_DIR}"/model-*-of-*.safetensors
  exit 0
fi

echo "[model] 下载 ${MODEL_ID} -> ${MODEL_DIR}"
"${PYTHON}" -m huggingface_hub.cli download "${MODEL_ID}" \
  --local-dir "${MODEL_DIR}" \
  --local-dir-use-symlinks False

echo "[model] 完成，分片:"
ls "${MODEL_DIR}"/model-*-of-*.safetensors
