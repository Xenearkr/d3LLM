#!/usr/bin/env bash
# 使用 4 张 GPU 并行评估 Dream-Coder（vanilla / d3LLM）在 HumanEval / MBPP 上的表现。
#
# 用法：
#   bash eval_scripts/dream_coder_parallel.sh vanilla humaneval
#   bash eval_scripts/dream_coder_parallel.sh vanilla mbpp
#   bash eval_scripts/dream_coder_parallel.sh d3llm humaneval
#   bash eval_scripts/dream_coder_parallel.sh d3llm mbpp
#
# 说明：
# - 依赖本仓库自带的 evalplus（路径：utils/utils_DreamCoder/code_eval/evalplus）
# - 会在 4 张卡上并行跑不同的 id_range，最终结果写入 evalplus_results/ 下的同一个 jsonl

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D3LLM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_KIND="${1:-}"
DATASET="${2:-}"

if [[ -z "$MODEL_KIND" || -z "$DATASET" ]]; then
  echo "用法: $0 <模型类型> <数据集>"
  echo "  模型类型: vanilla | d3llm"
  echo "  数据集: humaneval | mbpp"
  exit 1
fi

if [[ "$MODEL_KIND" != "vanilla" && "$MODEL_KIND" != "d3llm" ]]; then
  echo "错误: 模型类型须为 vanilla 或 d3llm"
  exit 1
fi

if [[ "$DATASET" != "humaneval" && "$DATASET" != "mbpp" ]]; then
  echo "错误: 数据集须为 humaneval 或 mbpp"
  exit 1
fi

if [[ "$HF_ALLOW_CODE_EVAL" != "1" ]]; then
  echo "请先设置: export HF_ALLOW_CODE_EVAL=1"
  echo "（EvalPlus 会执行生成代码，需显式允许）"
  exit 1
fi

cd "$D3LLM_ROOT/utils/utils_DreamCoder/code_eval/evalplus"

# 选择模型
if [[ "$MODEL_KIND" == "vanilla" ]]; then
  MODEL_NAME="Dream-org/Dream-Coder-v0-Instruct-7B"
else
  MODEL_NAME="d3LLM/d3LLM_Dream_Coder"
fi

echo "使用模型: $MODEL_NAME, 数据集: $DATASET，在 4 张 GPU 上并行评估"

# 根据数据集划分 id_range（含头不含尾）
if [[ "$DATASET" == "humaneval" ]]; then
  # 总 164 题 -> 4 段：0-41, 41-82, 82-123, 123-164
  RANGES=(
    "[0,41]"
    "[41,82]"
    "[82,123]"
    "[123,164]"
  )
else
  # MBPP Plus 总 399 题 -> 简单 4 段：0-100, 100-200, 200-300, 300-399
  RANGES=(
    "[0,100]"
    "[100,200]"
    "[200,300]"
    "[300,399]"
  )
fi

# 并行跑 4 段
PIDS=()
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$i" \
  PYTHONPATH=. \
  python -m evalplus.evaluate \
    "$DATASET" \
    --model "$MODEL_NAME" \
    --trust_remote_code=True \
    --backend dllm \
    --temperature 0.1 \
    --max_new_tokens 512 \
    --diffusion_steps 512 \
    --id_range "${RANGES[$i]}" \
    > "evalplus_parallel_${MODEL_KIND}_${DATASET}_gpu${i}.log" 2>&1 &

  PIDS+=($!)
  echo "GPU $i: 启动 id_range=${RANGES[$i]} 的任务，PID=${PIDS[-1]}"
done

echo "已启动所有并行任务，等待完成..."

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "PID $pid 任务失败"
    FAIL=1
  fi
done

if [[ "$FAIL" -ne 0 ]]; then
  echo "部分任务失败，请检查 evalplus_parallel_*.log 日志"
  exit 1
fi

echo "所有并行任务完成。结果已写入 evalplus_results/${DATASET}/ 下对应的 jsonl 文件。"

