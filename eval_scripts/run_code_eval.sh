#!/usr/bin/env bash
# 在 HumanEval / MBPP 上评估现有 dLLM
# 用法: bash run_code_eval.sh <模型> <数据集>
#   模型: d3llm_dream | d3llm_llada | vanilla_dream | vanilla_llada
#   数据集: humaneval | mbpp
# 需设置 HF_ALLOW_CODE_EVAL=1（脚本会检查并提示）

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D3LLM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="${1:-}"
DATASET="${2:-}"

if [[ -z "$MODEL" || -z "$DATASET" ]]; then
    echo "用法: $0 <模型> <数据集>"
    echo "  模型: d3llm_dream | d3llm_llada | vanilla_dream | vanilla_llada"
    echo "  数据集: humaneval | mbpp"
    exit 1
fi

if [[ "$DATASET" != "humaneval" && "$DATASET" != "mbpp" ]]; then
    echo "错误: 数据集须为 humaneval 或 mbpp"
    exit 1
fi

if [[ "$HF_ALLOW_CODE_EVAL" != "1" ]]; then
    echo "请先设置: export HF_ALLOW_CODE_EVAL=1"
    echo "（HumanEval/MBPP 会执行生成代码，需显式允许）"
    exit 1
fi

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

case "$MODEL" in
    d3llm_dream|vanilla_dream)
        # Dream 系: lm_eval, tasks 为 humaneval_instruct / mbpp_instruct
        TASK="${DATASET}_instruct"
        NUM_FEWSHOT=0
        [[ "$DATASET" == "mbpp" ]] && NUM_FEWSHOT=4
        cd "$D3LLM_ROOT/utils/utils_Dream/eval_instruct"
        if [[ "$MODEL" == "d3llm_dream" ]]; then
            PYTHONPATH=. accelerate launch --main_process_port 46666 -m lm_eval --model diffllm \
                --model_args torch_compile=False,pretrained=d3LLM/d3LLM_Dream,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.,alg=entropy_threshold,dParallel=False,threshold=0.5,generation_method=generation_multi_block,block_add_threshold=0.1,decoded_token_threshold=0.95,block_length=32,cache_delay_iter=10000,early_stop=True \
                --tasks "$TASK" --device cuda --batch_size 1 --num_fewshot "$NUM_FEWSHOT" \
                --output_path ./eval_tmp/code_eval_"$MODEL"_"$DATASET" --log_samples --confirm_run_unsafe_code --apply_chat_template
        else
            PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval --model diffllm \
                --model_args torch_compile=False,pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.1,top_p=0.9,alg=entropy,dParallel=False \
                --tasks "$TASK" --device cuda --batch_size 1 --num_fewshot "$NUM_FEWSHOT" \
                --output_path ./eval_tmp/code_eval_"$MODEL"_"$DATASET" --log_samples --confirm_run_unsafe_code --apply_chat_template
        fi
        echo "Dream 评估完成。结果见: $D3LLM_ROOT/utils/utils_Dream/eval_instruct/eval_tmp/code_eval_${MODEL}_${DATASET}/"
        ;;
    d3llm_llada|vanilla_llada)
        # LLaDA 系: eval_llada.py
        LENGTH=256
        BLOCK_LENGTH=32
        STEPS=256
        NUM_FEWSHOT=0
        [[ "$DATASET" == "mbpp" ]] && NUM_FEWSHOT=3

        # LLaDA 系与论文/基线一致：使用 humaneval（base），不用 humaneval_instruct；pass@1 以 postprocess_code_humaneval.py 输出为准
        TASK="$DATASET"

        if [[ "$MODEL" == "d3llm_llada" ]]; then
            MODEL_PATH="d3LLM/d3LLM_LLaDA"
            OUTPUT_DIR="d3llm"
        else
            MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct"
            OUTPUT_DIR="GSAI-ML"
        fi
        METHOD_NAME_ENCODED="${MODEL_PATH//\//__}"
        cd "$D3LLM_ROOT/utils/utils_LLaDA"
        accelerate launch --main_process_port 29600 eval_llada.py --tasks "$TASK" --num_fewshot "$NUM_FEWSHOT" \
            --confirm_run_unsafe_code --model llada_dist \
            --model_args model_path="$MODEL_PATH",gen_length=$LENGTH,steps=$STEPS,block_length=$BLOCK_LENGTH,show_speed=True,task="$TASK" \
            --output_path "evals_results/${OUTPUT_DIR}/${DATASET}-run_code_eval-${MODEL}-ns${NUM_FEWSHOT}-${LENGTH}" --log_samples
        latest_jsonl=$(find "evals_results/${OUTPUT_DIR}/${DATASET}-run_code_eval-${MODEL}-ns${NUM_FEWSHOT}-${LENGTH}/${METHOD_NAME_ENCODED}" -name "samples_${DATASET}_*.jsonl" -type f 2>/dev/null | head -n 1)
        if [[ -n "$latest_jsonl" ]]; then
            if [[ "$DATASET" == "humaneval" ]]; then
                python postprocess_code_humaneval.py "$latest_jsonl"
            else
                python postprocess_code_mbpp.py "$latest_jsonl"
            fi
        else
            echo "未找到 samples jsonl，请到 evals_results/${OUTPUT_DIR}/ 下查看"
        fi
        echo "LLaDA 评估完成。结果见: $D3LLM_ROOT/utils/utils_LLaDA/evals_results/${OUTPUT_DIR}/${DATASET}-run_code_eval-${MODEL}-ns${NUM_FEWSHOT}-${LENGTH}/"
        ;;
    *)
        echo "错误: 未知模型 $MODEL"
        echo "  可选: d3llm_dream | d3llm_llada | vanilla_dream | vanilla_llada"
        exit 1
        ;;
esac
