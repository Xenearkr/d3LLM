#!/usr/bin/env bash
# 兼容包装：请优先使用
#   bash eval_scripts/run_code_eval.sh diffllm <模型> <task> [m] [s] [t]
#
# 本脚本保留旧参数顺序：仅传 task 与可选 m/s/t，模型来自 PRETRAINED 或 MERGED_MODEL_PATH。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MERGED_MODEL_PATH="${MERGED_MODEL_PATH:-${REPO_ROOT}/output_model/merged_d3LLM_DREAM_5742}"
PRETRAINED_ARG="${PRETRAINED:-$MERGED_MODEL_PATH}"

if [ "$#" -lt 1 ]; then
  echo "用法: $0 <task_name> [max_new_tokens] [diffusion_steps] [threshold]" >&2
  echo "模型: 设置 PRETRAINED 或 MERGED_MODEL_PATH（当前: ${PRETRAINED_ARG}）" >&2
  echo "推荐: bash eval_scripts/run_code_eval.sh diffllm <模型> <task> [m] [s] [t]" >&2
  exit 1
fi

exec bash "${REPO_ROOT}/eval_scripts/run_code_eval.sh" diffllm "$PRETRAINED_ARG" "$1" "${2:-256}" "${3:-256}" "${4:-0.4}"
