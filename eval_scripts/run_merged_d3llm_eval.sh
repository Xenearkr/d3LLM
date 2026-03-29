#!/usr/bin/env bash

# 用 d3LLM-Dream（合并后的本地目录或 Hugging Face 模型 ID）在任意任务上跑 lm_eval
# 输入：任务名 + max_new_tokens + diffusion_steps + threshold
# 用法示例（在项目根或 eval_scripts 下运行）:
#   bash eval_scripts/run_merged_d3llm_eval.sh gsm8k_cot_zeroshot 256 256 0.4
#   PRETRAINED=d3LLM/d3LLM_Dream_Coders bash eval_scripts/run_merged_d3llm_eval.sh humaneval_instruct 256 256 0.4
#
# 模型来源（二选一，PRETRAINED 优先）:
#   PRETRAINED=org/model   # Hub，不做本地 config 检查
#   MERGED_MODEL_PATH=...  # 默认仍为 output_model/merged_d3LLM_DREAM_5742
#
# 输出：平均 TPF（tokens per forward）和准确率（exact_match, flexible-extract）
#
# 进程数：默认 ACCELERATE_NUM_PROCESSES=1（单卡）。多卡数据并行时：
#   ACCELERATE_NUM_PROCESSES=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash eval_scripts/run_merged_d3llm_eval.sh ...

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "用法: $0 <task_name> [max_new_tokens] [diffusion_steps] [threshold]" >&2
  exit 1
fi

TASK_NAME="$1"
MAX_NEW_TOKENS="${2:-256}"
DIFFUSION_STEPS="${3:-256}"
THRESHOLD="${4:-0.4}"

# 项目根目录
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MERGED_MODEL_PATH="${MERGED_MODEL_PATH:-${REPO_ROOT}/output_model/merged_d3LLM_DREAM_5742}"
# PRETRAINED 优先：可为 Hub 上的 org/model，或任意本地/缓存路径（与 lm_eval pretrained= 一致）
PRETRAINED_ARG="${PRETRAINED:-$MERGED_MODEL_PATH}"

if [[ -d "${PRETRAINED_ARG}" ]] && [[ ! -f "${PRETRAINED_ARG}/config.json" ]]; then
  echo "错误: 本地目录 '${PRETRAINED_ARG}' 下未找到 config.json。" >&2
  exit 1
fi

NUM_PROC="${ACCELERATE_NUM_PROCESSES:-1}"
# 未显式设置 CUDA_VISIBLE_DEVICES 时：单进程只用 GPU0，避免 4 进程各加载一份 7B 长时间无新日志
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  if [ "$NUM_PROC" -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=0
  else
    export CUDA_VISIBLE_DEVICES=0,1,2,3
  fi
fi
export HF_ALLOW_CODE_EVAL=1

EVAL_ROOT="${REPO_ROOT}/utils/utils_Dream/eval_instruct"
cd "${EVAL_ROOT}"

OUTPUT_DIR="${EVAL_ROOT}/eval_tmp/merged_${TASK_NAME}_m${MAX_NEW_TOKENS}_s${DIFFUSION_STEPS}_t${THRESHOLD}"
mkdir -p "${OUTPUT_DIR}"

LOG_FILE="${OUTPUT_DIR}/run.log"

echo "运行任务: ${TASK_NAME}"
echo "max_new_tokens = ${MAX_NEW_TOKENS}, diffusion_steps = ${DIFFUSION_STEPS}, threshold = ${THRESHOLD}"
echo "pretrained: ${PRETRAINED_ARG}"
echo "accelerate num_processes: ${NUM_PROC}  (设置 ACCELERATE_NUM_PROCESSES 可改)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "结果目录: ${OUTPUT_DIR}"

accelerate launch --num_processes "${NUM_PROC}" --main_process_port 46667 -m lm_eval \
  --model diffllm \
  --model_args "torch_compile=False,pretrained=${PRETRAINED_ARG},trust_remote_code=True,max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},dtype=bfloat16,temperature=0.,alg=entropy_threshold,dParallel=False,threshold=${THRESHOLD},generation_method=generation_multi_block,block_add_threshold=0.1,decoded_token_threshold=0.95,block_length=32,cache_delay_iter=1,refresh_interval=10000,early_stop=True" \
  --tasks "${TASK_NAME}" \
  --device cuda \
  --batch_size 1 \
  --num_fewshot 0 \
  --output_path "${OUTPUT_DIR}" \
  --log_samples \
  --confirm_run_unsafe_code \
  --apply_chat_template 2>&1 | tee "${LOG_FILE}"

echo
echo "评测完成，正在解析 TPF 和准确率..."

# 提取最后一次统计中的 Tokens per forward
TPF_LINE="$(grep 'Tokens per forward' "${LOG_FILE}" | tail -n 1 || true)"
if [ -z "${TPF_LINE}" ]; then
  echo "警告: 未在日志中找到 'Tokens per forward' 行，无法解析 TPF。" >&2
  AVG_TPF="N/A"
else
  # 形如: '  Tokens per forward: 4.73'
  AVG_TPF="$(echo "${TPF_LINE}" | awk '{print $4}')"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

ACC_OUTPUT="$(
  "${PYTHON_BIN}" - << 'PY' "${OUTPUT_DIR}" "${TASK_NAME}"
import json
import os
import sys
import glob

output_dir, task_name = sys.argv[1], sys.argv[2]

pattern = os.path.join(output_dir, "**", "results_*.json")
files = glob.glob(pattern, recursive=True)
if not files:
    print("N/A")
    sys.exit(0)

files.sort(key=os.path.getmtime)
results_path = files[-1]

with open(results_path, "r", encoding="utf-8") as f:
    data = json.load(f)

task_res = data.get("results", {}).get(task_name)
if not task_res:
    print("N/A")
    sys.exit(0)

acc = None
for k, v in task_res.items():
    if k.startswith("exact_match") and "flexible-extract" in k:
        acc = v
        break

if acc is None and "exact_match" in task_res:
    acc = task_res["exact_match"]

if acc is None:
    print("N/A")
else:
    print(f"{acc:.6f}")
PY
)"

echo "=========== 汇总 =========="
echo "任务: ${TASK_NAME}"
echo "max_new_tokens: ${MAX_NEW_TOKENS}"
echo "diffusion_steps: ${DIFFUSION_STEPS}"
echo "threshold: ${THRESHOLD}"
echo "平均 TPF (Tokens per forward): ${AVG_TPF}"
echo "准确率 (exact_match, flexible-extract): ${ACC_OUTPUT}"
echo "结果 JSON 和样本日志保存在: ${OUTPUT_DIR}"

