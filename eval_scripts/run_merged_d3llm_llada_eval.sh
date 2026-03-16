#!/usr/bin/env bash

# 用合并后的自训练 d3LLM-LLaDA 在任意 LLaDA 任务上跑 eval_llada.py
# 输入：任务名 + max_new_tokens + diffusion_steps + threshold
#
# 用法示例（在项目根或 eval_scripts 下运行）:
#   bash eval_scripts/run_merged_d3llm_llada_eval.sh gsm8k_cot 256 256 0.5
#
# 输出：eval_llada 的完整结果 JSON；脚本末尾会打印该任务的主指标。

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "用法: $0 <task_name> [max_new_tokens] [diffusion_steps] [threshold]" >&2
  echo "  示例: $0 gsm8k_cot 256 256 0.5" >&2
  exit 1
fi

TASK_NAME="$1"
MAX_NEW_TOKENS="${2:-256}"
DIFFUSION_STEPS="${3:-256}"
THRESHOLD="${4:-0.5}"

# 项目根目录
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# 合并后的自训练 LLaDA 模型路径（可通过环境变量覆盖）
MERGED_LLaDA_MODEL_PATH="${MERGED_LLaDA_MODEL_PATH:-${REPO_ROOT}/output_model/merged_d3LLM_LLaDA_8664}"

if [ ! -f "${MERGED_LLaDA_MODEL_PATH}/config.json" ]; then
  echo "错误: 在 MERGED_LLaDA_MODEL_PATH='${MERGED_LLaDA_MODEL_PATH}' 下未找到 config.json，请确认模型路径是否正确。" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

LLADA_ROOT="${REPO_ROOT}/utils/utils_LLaDA"
cd "${LLADA_ROOT}"

OUTPUT_DIR="${LLADA_ROOT}/evals_results/d3llm_merged/${TASK_NAME}-merged_d3LLM_LLaDA_m${MAX_NEW_TOKENS}_s${DIFFUSION_STEPS}_t${THRESHOLD}"
mkdir -p "${OUTPUT_DIR}"

echo "运行 LLaDA 任务: ${TASK_NAME}"
echo "max_new_tokens = ${MAX_NEW_TOKENS}, diffusion_steps = ${DIFFUSION_STEPS}, threshold = ${THRESHOLD}"
echo "模型路径: ${MERGED_LLaDA_MODEL_PATH}"
echo "结果目录: ${OUTPUT_DIR}"

# 与现有 llada_eval.sh 一致：使用 eval_llada.py + llada_dist
accelerate launch --main_process_port 29610 eval_llada.py \
  --tasks "${TASK_NAME}" \
  --num_fewshot 0 \
  --confirm_run_unsafe_code \
  --model llada_dist \
  --model_args "model_path=${MERGED_LLaDA_MODEL_PATH},gen_length=${MAX_NEW_TOKENS},steps=${DIFFUSION_STEPS},block_length=32,show_speed=True,task=${TASK_NAME},remasking=low_confidence,threshold=${THRESHOLD}" \
  --output_path "${OUTPUT_DIR}" \
  --log_samples

echo
echo "=========== 简要结果 =========="

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" - << 'PY' "${OUTPUT_DIR}" "${TASK_NAME}"
import json, os, sys, glob

out_dir, task = sys.argv[1], sys.argv[2]
pattern = os.path.join(out_dir, "**", "results_*.json")
files = glob.glob(pattern, recursive=True)
if not files:
    print("未找到 results_*.json，请到目录手动查看:", out_dir)
    sys.exit(0)

files.sort(key=os.path.getmtime)
path = files[-1]
print(f"使用结果文件: {os.path.relpath(path, out_dir)}")

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

res = data.get("results", {}).get(task)
if not res:
    print(f"未在 results 中找到任务 {task}，完整内容：")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    sys.exit(0)

print("任务:", task)
for k, v in res.items():
    if isinstance(v, (int, float)):
        print(f"  {k}: {v:.6f}")
    else:
        print(f"  {k}: {v}")
PY

echo "完整结果和样本日志保存在: ${OUTPUT_DIR}"