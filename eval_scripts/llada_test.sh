#!/usr/bin/env bash
set -euo pipefail

# 只做两组对比：
# 1) LLaDA baseline
# 2) 你的新训练模型（可改 NEW_MODEL_PATH）

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

cd /home/u-chenx/chenx/d3LLM/utils/utils_LLaDA

TASK="gsm8k_cot_zeroshot"
NUM_FEWSHOT=0
GEN_LENGTH=256
STEPS=256
BLOCK_LENGTH=32
BATCH_SIZE=4

BASELINE_MODEL="GSAI-ML/LLaDA-8B-Instruct"
NEW_MODEL_PATH="/home/u-chenx/Codes/d3LLM/output_model/d3LLM_LLaDA_merged_0331_031307"

echo "==== [1/2] Baseline: ${BASELINE_MODEL} ===="
accelerate launch --main_process_port 29610 eval_llada.py \
  --tasks "${TASK}" \
  --num_fewshot "${NUM_FEWSHOT}" \
  --confirm_run_unsafe_code \
  --model llada_dist \
  --model_args model_path="${BASELINE_MODEL}",gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},show_speed=True,task="${TASK}" \
  --batch_size "${BATCH_SIZE}" \
  --output_path "evals_results/llada_baseline_gsm8k_cot_zeroshot"

echo "==== [2/2] New model: ${NEW_MODEL_PATH} ===="
accelerate launch --main_process_port 29611 eval_llada.py \
  --tasks "${TASK}" \
  --num_fewshot "${NUM_FEWSHOT}" \
  --confirm_run_unsafe_code \
  --model llada_dist \
  --model_args model_path="${NEW_MODEL_PATH}",gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},show_speed=True,task="${TASK}" \
  --batch_size "${BATCH_SIZE}" \
  --output_path "evals_results/llada_newmodel_gsm8k_cot_zeroshot"

echo "Done. 对比两个 output_path 下的结果即可。"