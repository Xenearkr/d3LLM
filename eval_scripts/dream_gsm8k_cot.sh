# 用 screen 跑并输出到同目录 log（在项目根执行）:
#   screen -dmS dream_gsm8k_cot bash -c 'bash eval_scripts/dream_gsm8k_cot.sh > eval_scripts/dream_gsm8k_cot.log 2>&1'
# 查看: tail -f eval_scripts/dream_gsm8k_cot.log  恢复: screen -r dream_gsm8k_cot

# 合并后的自训练模型路径（用 merge_lora_dream.py 得到），按需修改；默认用项目根下 output_model/merged_d3LLM_DREAM_5742
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MERGED_MODEL_PATH="${MERGED_MODEL_PATH:-${REPO_ROOT}/output_model/merged_d3LLM_DREAM_5742}"

# 这个因为环境问题跑不了
Qwen2.5-7B-Instruct, gsm8k_cot_zeroshot
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HF_ALLOW_CODE_EVAL=1
cd "${REPO_ROOT}/utils/lm-evaluation-harness"
PYTHONPATH="${REPO_ROOT}/utils/lm-evaluation-harness:$PYTHONPATH" \
accelerate launch -m lm_eval \
    --model hf \
    --model_args "pretrained=Qwen/Qwen2.5-7B-Instruct,temperature=0.0" \
    --tasks gsm8k_cot_zeroshot \
    --num_fewshot 0 \
    --batch_size 32 \
    --output_path evals_results/gsm8k_cot_zeroshot   \
    --log_samples \
    --confirm_run_unsafe_code \
    --gen_kwargs do_sample=False,max_gen_toks=256


## Vanilla Dream, TPF=1.0:
cd "${REPO_ROOT}/utils/utils_Dream/eval_instruct"
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args torch_compile=False,pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
    --tasks gsm8k_cot_zeroshot \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path eval_tmp/gsm8k_cot_zeroshot \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template


# Fast-dLLM Dream (dual cache):
cd "${REPO_ROOT}/utils/utils_Dream/eval_instruct"
accelerate launch --main_process_port 12334 -m lm_eval --model diffllm --model_args torch_compile=False,pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=256,diffusion_steps=8,dtype=bfloat16,temperature=0.,alg=confidence_threshold,threshold=0.9,generation_method=Fast_dllm_v1,use_cache=True,dual_cache=True,block_length=32 --tasks gsm8k_cot_zeroshot --device cuda --batch_size 1 --num_fewshot 0 --output_path ./eval_tmp/fast_dllm_dual_cache --log_samples --confirm_run_unsafe_code --apply_chat_template



# dParallel-Dream, TPF=1.0:
cd "${REPO_ROOT}/utils/utils_Dream/eval_instruct"
accelerate launch --main_process_port 12334 -m lm_eval --model diffllm --model_args torch_compile=False,pretrained=Zigeng/dParallel_Dream_7B_Instruct,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.1,top_p=0.9,alg=entropy,dParallel=False --tasks gsm8k_cot_zeroshot --device cuda --batch_size 1 --num_fewshot 0 --output_path ./eval_tmp/multi_block_cot --log_samples --confirm_run_unsafe_code --apply_chat_template


# dParallel-Dream, entropy-threshold=0.45:
cd "${REPO_ROOT}/utils/utils_Dream/eval_instruct"
accelerate launch --main_process_port 12334 -m lm_eval --model diffllm --model_args torch_compile=False,pretrained=Zigeng/dParallel_Dream_7B_Instruct,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.,alg="entropy_threshold",dParallel=True,threshold=0.45 --tasks gsm8k_cot_zeroshot --device cuda --batch_size 1 --num_fewshot 0 --output_path ./eval_tmp/entropy_threshold_0.45 --log_samples --confirm_run_unsafe_code --apply_chat_template


# d3LLM-Dream, TPF=1.0:
cd "${REPO_ROOT}/utils/utils_Dream/eval_instruct"
accelerate launch --main_process_port 46666 -m lm_eval --model diffllm --model_args pretrained=d3LLM/d3LLM_Dream,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.1,top_p=0.9,alg=entropy,dParallel=False --tasks gsm8k_cot_zeroshot --device cuda --batch_size 1 --num_fewshot 0 --output_path ./eval_tmp/multi_block_cot --log_samples --confirm_run_unsafe_code --apply_chat_template


# d3LLM-Dream: generate_multi_block (no delay), threshold=0.4:
cd "${REPO_ROOT}/utils/utils_Dream/eval_instruct"
accelerate launch --main_process_port 46666 -m lm_eval --model diffllm --model_args torch_compile=True,pretrained=d3LLM/d3LLM_Dream,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.,alg=entropy_threshold,dParallel=False,threshold=0.4,generation_method=generation_multi_block,block_add_threshold=0.1,decoded_token_threshold=0.95,block_length=32,cache_delay_iter=10000 --tasks gsm8k_cot_zeroshot --device cuda --batch_size 1 --num_fewshot 0 --output_path ./eval_tmp/multi_block_cot --log_samples --confirm_run_unsafe_code --apply_chat_template

# d3LLM-Dream: generate_multi_block_kv_cache, delay=1:
cd "${REPO_ROOT}/utils/utils_Dream/eval_instruct"
accelerate launch --main_process_port 46666 -m lm_eval --model diffllm --model_args torch_compile=False,pretrained=d3LLM/d3LLM_Dream,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.,alg=entropy_threshold,dParallel=False,threshold=0.4,generation_method=generation_multi_block,block_add_threshold=0.1,decoded_token_threshold=0.95,block_length=32,cache_delay_iter=1,refresh_interval=10000,early_stop=True --tasks gsm8k_cot_zeroshot --device cuda --batch_size 1 --num_fewshot 0 --output_path ./eval_tmp/multi_block_cot --log_samples --confirm_run_unsafe_code --apply_chat_template


# 合并后的自训练 d3LLM-Dream（merge_lora_dream.py 产出），TPF=1.0:
cd "${REPO_ROOT}/utils/utils_Dream/eval_instruct"
accelerate launch --main_process_port 46666 -m lm_eval --model diffllm --model_args torch_compile=False,pretrained=/home/u-chenx/chenx/d3LLM-new/output_model/merged_d3LLM_DREAM_5742,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.,alg=entropy_threshold,dParallel=False,threshold=0.4,generation_method=generation_multi_block,block_add_threshold=0.1,decoded_token_threshold=0.95,block_length=32,cache_delay_iter=1,refresh_interval=10000,early_stop=True --tasks gsm8k_cot_zeroshot --device cuda --batch_size 1 --num_fewshot 0 --output_path ./eval_tmp/multi_block_cot --log_samples --confirm_run_unsafe_code --apply_chat_template

