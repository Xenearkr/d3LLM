#!/bin/bash

# 单机 4 卡并行生成轨迹（模拟 SLURM 环境）

NUM_GPUS=4
STEPS=512
GEN_LENGTH=512
BLOCK_LENGTH=32
OUTPUT_DIR="trajectory_data_1node_4gpu"
MAX_DATA_NUM=4   # -1 表示不限数据量；可以改成比如 10000 先测试

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# distill_1_data_prepare -> d3llm_DREAM -> d3llm -> repo_root
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

for RANK in 0 1 2 3; do
  echo "Launching rank ${RANK}/${NUM_GPUS} on GPU ${RANK}..."

  CUDA_VISIBLE_DEVICES=${RANK} \
  SLURM_PROCID=${RANK} \
  SLURM_LOCALID=${RANK} \
  SLURM_NTASKS=${NUM_GPUS} \
  "${PYTHON_BIN}" d3llm/d3llm_DREAM/distill_1_data_prepare/d3llm_dream_generate_multinode.py \
    --num_gpus ${NUM_GPUS} \
    --steps ${STEPS} \
    --gen_length ${GEN_LENGTH} \
    --block_length ${BLOCK_LENGTH} \
    --output_dir ${OUTPUT_DIR} \
    --max_data_num ${MAX_DATA_NUM} \
    --trajectory_one_step &
done

wait
echo "All 4 GPU processes finished. Output saved to ${OUTPUT_DIR}"

# 使用方式：
# cd /home/u-chenx/chenx/d3LLM/d3llm/d3llm_DREAM/distill_1_data_prepare
# chmod +x run_generate_1node_4gpu.sh
# ./run_generate_1node_4gpu.sh