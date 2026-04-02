#!/usr/bin/env bash
# 兼容包装：请优先使用
#   bash eval_scripts/run_code_eval.sh diffllm <模型> <task> [m] [s] [t]
#
# 本脚本参数顺序：<task_name> [max_new_tokens] [diffusion_steps] [threshold] [merged_model]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

DEFAULT_MERGED_MODEL_PATH="${REPO_ROOT}/output_model/merged_d3LLM_DREAM_5742"
MERGED_MODEL_PATH="${5:-${DEFAULT_MERGED_MODEL_PATH}}"

if [ "$#" -lt 1 ]; then
  echo "用法: $0 <task_name> [max_new_tokens] [diffusion_steps] [threshold] [merged_model]" >&2
  echo "模型: merged_model 通过参数传递（未传则使用默认: ${DEFAULT_MERGED_MODEL_PATH}）" >&2
  echo "推荐: bash eval_scripts/run_code_eval.sh diffllm <模型> <task> [m] [s] [t]" >&2
  exit 1
fi

exec bash "${REPO_ROOT}/eval_scripts/run_code_eval.sh" diffllm "${MERGED_MODEL_PATH}" "$1" "${2:-256}" "${3:-256}" "${4:-0.4}"
