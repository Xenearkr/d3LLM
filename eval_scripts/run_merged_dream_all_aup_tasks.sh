#!/usr/bin/env bash
# 在「Dream 合并权重」上依次评测 AUP_leaderboard/data_dream.yaml 中五个顶层任务：
#   GSM8K-CoT, MATH, MBPP-Instruct, HumanEval-Instruct, Long-GSM8K
#
# 实现方式：对每条任务调用 eval_scripts/run_code_eval.sh 中已有 recipe（与 dream_gsm8k_cot.sh
# 里合并模型 + multiblock + KV delay 的写法一致；GSM8K-CoT 使用 merged_mblock_kv_delay1）。
#
# 用法（在仓库根目录 d3LLM 下）:
#   bash eval_scripts/run_merged_dream_all_aup_tasks.sh /path/to/merged_dream_model
#   MERGED_MODEL_PATH=/path/to/merged bash eval_scripts/run_merged_dream_all_aup_tasks.sh
#
# 可选环境变量:
#   CUDA_VISIBLE_DEVICES  默认 0,1,2,3
#   CONTINUE_ON_ERROR=1   某条失败也继续跑后续任务（默认遇错即停）

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MERGED_MODEL_PATH="${1:-${MERGED_MODEL_PATH:-}}"

if [[ -z "${MERGED_MODEL_PATH}" ]]; then
  echo "用法: $0 <merged_dream_model_dir>" >&2
  echo "  或: MERGED_MODEL_PATH=<dir> $0" >&2
  echo "合并目录须为 merge_lora_dream.py 产出的 Dream 权重（含 DreamConfig）。" >&2
  exit 1
fi

if [[ ! -f "${MERGED_MODEL_PATH}/config.json" ]]; then
  echo "错误: 未找到 ${MERGED_MODEL_PATH}/config.json" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"

# data_dream.yaml 顶层键 -> run_code_eval.sh recipe（第二参数统一覆盖为合并模型路径）
# GSM8K-CoT: 对齐 dream_gsm8k_cot.sh L74-76（multiblock + cache_delay_iter=1）
declare -a ROWS=(
  "GSM8K-CoT|dream_gsm8k_cot__merged_mblock_kv_delay1"
  "MATH|dream_math__d3llm_multiblock_delay2"
  "MBPP-Instruct|dream_mbpp__d3llm_multiblock_delay2"
  "HumanEval-Instruct|dream_humaneval__d3llm_multiblock_delay2"
  "Long-GSM8K|dream_long_gsm8k__d3llm_multiblock_kv_delay1"
)

echo "================================================================================"
echo "AUP data_dream 全任务评测（合并 Dream）"
echo "  REPO_ROOT=${REPO_ROOT}"
echo "  MERGED_MODEL_PATH=${MERGED_MODEL_PATH}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "================================================================================"

fail_any=0
for row in "${ROWS[@]}"; do
  IFS='|' read -r yaml_task recipe <<<"${row}"
  echo
  echo "---------- [${yaml_task}] -> recipe=${recipe} ----------"
  if bash "${REPO_ROOT}/eval_scripts/run_code_eval.sh" "${recipe}" "${MERGED_MODEL_PATH}"; then
    echo "[OK] ${yaml_task}"
  else
    echo "[FAIL] ${yaml_task} (recipe=${recipe})" >&2
    fail_any=1
    if [[ "${CONTINUE_ON_ERROR:-0}" != "1" ]]; then
      exit 1
    fi
  fi
done

if [[ "${fail_any}" -ne 0 ]]; then
  echo >&2 "部分任务失败；已设置 CONTINUE_ON_ERROR=1 时会跑完全部。"
  exit 1
fi

echo
echo "全部完成。"
