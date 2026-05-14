#!/usr/bin/env bash
# EvalPlus HumanEval+/MBPP+，经 OpenAI 兼容接口调本机 SGLang。
# 使用前需要先启动本机 SGLang，参考 README.md 中的安装/启动方法：
# 在和本项目不同的环境中：
# ```bash
# pip install uv
# uv pip install "sglang[all] @ git+https://github.com/sgl-project/sglang.git@refs/pull/20615/head#subdirectory=python"
# ```

# 然后启动 SGLang 服务：

# ```bash
# # d3LLM-LLaDA 可换成 d3LLM/d3LLM_Dream
# python -m sglang.launch_server \
#     --model d3LLM/d3LLM_LLaDA \
#     --trust-remote-code \
#     --attention-backend flashinfer \
#     --dllm-algorithm FullAttnMultiBlock \
#     --mem-fraction-static 0.8 \
#     --cuda-graph-max-bs 32

# 服务启动后，在不同终端中执行：
# bash eval_scripts/evalplus_by_sglang.sh humaneval
# bash eval_scripts/evalplus_by_sglang.sh mbpp

set -euo pipefail

_script_path="${BASH_SOURCE[0]}"
[[ "${_script_path}" != /* ]] && _script_path="$(pwd)/${_script_path}"
SCRIPT_DIR="$(cd "$(dirname "${_script_path}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
unset _script_path

EVALPLUS_ROOT="${REPO_ROOT}/utils/utils_DreamCoder/code_eval/evalplus"
[[ -f "${EVALPLUS_ROOT}/evalplus/evaluate.py" ]] || {
  echo "未找到 EvalPlus: ${EVALPLUS_ROOT}/evalplus/evaluate.py" >&2
  exit 1
}

TASK="${1:-}"
[[ "${TASK}" == "humaneval" || "${TASK}" == "mbpp" || "${TASK}" == "all" ]] || {
  echo "用法: $0 {humaneval|mbpp|all}" >&2
  exit 1
}

PYTHON="${PYTHON:-python3}"
SGLANG_OPENAI_BASE="${SGLANG_OPENAI_BASE:-http://127.0.0.1:30000/v1}"
BASE_URL="${SGLANG_OPENAI_BASE}"
BASE_HTTP="${BASE_URL%/v1}"
BASE_HTTP="${BASE_HTTP%/}"

export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
GREEDY="${GREEDY:-True}"
BS="${BS:-1}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/evalplus_results_sglang}"

resolve_model_name() {
  [[ -n "${SGLANG_SERVED_MODEL:-}" ]] && {
    echo "${SGLANG_SERVED_MODEL}"
    return
  }
  command -v curl >/dev/null || {
    echo "请设置 SGLANG_SERVED_MODEL 或安装 curl" >&2
    exit 1
  }
  local mi="${BASE_HTTP}/model_info" js
  js="$(curl -sS --fail --connect-timeout 2 --noproxy '*' "${mi}")" || {
    echo "无法访问 ${mi} ，检查 SGLang 与 SGLANG_OPENAI_BASE=${SGLANG_OPENAI_BASE}" >&2
    exit 1
  }
  if command -v jq >/dev/null 2>&1; then
    echo "${js}" | jq -r '.model_path // .served_model_name // empty'
  else
    "${PYTHON}" -c "import json,sys; d=json.loads(sys.argv[1]); print(d.get('model_path') or d.get('served_model_name') or '')" "${js}"
  fi
}

MODEL="$(resolve_model_name)"
[[ -n "${MODEL}" && "${MODEL}" != "null" ]] || {
  echo "请 export SGLANG_SERVED_MODEL=<与 chat 接口 model 一致>" >&2
  exit 1
}

echo "[evalplus+sglang] BASE_URL=${BASE_URL} MODEL=${MODEL} RESULT_ROOT=${RESULT_ROOT}"

# 仅连本机 API：去掉 SOCKS/HTTP 代理，否则 httpx 走 SOCKS 且未装 socksio 会在 OpenAI() 处 ImportError
_evalplus_run() {
  export PYTHONPATH="${EVALPLUS_ROOT}"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
  if "${PYTHON}" -P -c "pass" 2>/dev/null; then
    "${PYTHON}" -P -m evalplus.evaluate "$@"
  else
    (cd "${REPO_ROOT}" && "${PYTHON}" -m evalplus.evaluate "$@")
  fi
}

run_one() {
  echo "========== ${1} =========="
  _evalplus_run "${1}" \
    --model "${MODEL}" --backend openai --base_url "${BASE_URL}" \
    --trust_remote_code True --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" --greedy "${GREEDY}" --bs "${BS}" \
    --root "${RESULT_ROOT}"
}

case "${TASK}" in
  humaneval) run_one humaneval ;;
  mbpp) run_one mbpp ;;
  all) run_one humaneval && run_one mbpp ;;
esac

echo "[evalplus+sglang] 完成 -> ${RESULT_ROOT}/<dataset>/"
