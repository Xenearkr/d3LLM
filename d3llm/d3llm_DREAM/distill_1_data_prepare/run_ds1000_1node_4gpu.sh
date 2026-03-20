#!/bin/bash

# 单机 4 卡并行生成轨迹（模拟 SLURM 环境）
set -euo pipefail

NUM_GPUS=4
STEPS=512
GEN_LENGTH=512
BLOCK_LENGTH=32
OUTPUT_DIR="trajectory_data_1node_4gpu_ds1000"
MAX_DATA_NUM=16   # -1 表示不限数据量；可以改成比如 10000 先测试
TRAJECTORY_ONE_STEP=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# distill_1_data_prepare -> d3llm_DREAM -> d3llm -> repo_root
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

PIDS=()
INTERRUPTED=0

cleanup() {
  local sig="${1:-EXIT}"
  # 防止重复清理
  trap - INT TERM EXIT

  if [ "${#PIDS[@]}" -gt 0 ]; then
    echo "[cleanup] Caught ${sig}, terminating ${#PIDS[@]} worker(s)..."
    # 先尝试优雅退出
    for pid in "${PIDS[@]}"; do
      kill -TERM "${pid}" 2>/dev/null || true
    done
    sleep 1
    # 仍存活则强杀
    for pid in "${PIDS[@]}"; do
      kill -KILL "${pid}" 2>/dev/null || true
    done
  fi

  # 兜底：按脚本特征清理残留 Python 进程
  pkill -f "ds1000_multinode.py.*--output_dir ${OUTPUT_DIR}" 2>/dev/null || true
  pkill -f "ds1000_partly.py" 2>/dev/null || true
}

on_interrupt() {
  INTERRUPTED=1
  cleanup "INT/TERM"
  exit 130
}

trap 'on_interrupt' INT TERM
trap 'cleanup "EXIT"' EXIT

for RANK in 0 1 2 3; do
  echo "Launching rank ${RANK}/${NUM_GPUS} on GPU ${RANK}..."

  CUDA_VISIBLE_DEVICES=${RANK} \
  SLURM_PROCID=${RANK} \
  SLURM_LOCALID=${RANK} \
  SLURM_NTASKS=${NUM_GPUS} \
  "${PYTHON_BIN}" d3llm/d3llm_DREAM/distill_1_data_prepare/ds1000_multinode.py \
    --num_gpus ${NUM_GPUS} \
    --steps ${STEPS} \
    --gen_length ${GEN_LENGTH} \
    --block_length ${BLOCK_LENGTH} \
    --output_dir ${OUTPUT_DIR} \
    --max_data_num ${MAX_DATA_NUM} \
    $( [ "${TRAJECTORY_ONE_STEP}" = "true" ] && echo "--trajectory_one_step" ) &
  PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
  # wait 会返回非 0；在 set -e 下需接管错误，不然无法继续清理/汇总
  if ! wait "${pid}"; then
    echo "[error] Worker PID ${pid} failed."
    INTERRUPTED=1
  fi
done

if [ "${INTERRUPTED}" -eq 0 ]; then
  echo "All ${NUM_GPUS} GPU processes finished. Output saved to ${OUTPUT_DIR}"
else
  echo "Generation exited with interruption/failures. Check logs and partial outputs in ${OUTPUT_DIR}"
  exit 1
fi

# 使用方式：
# cd /home/u-chenx/chenx/d3LLM/d3llm/d3llm_DREAM/distill_1_data_prepare
# chmod +x run_generate_1node_4gpu.sh
# ./run_generate_1node_4gpu.sh