#!/usr/bin/env bash
# 汇总 eval_scripts 下 dream_* / dream-coder.sh / llada_* 中的评测命令，与参考脚本逐行一致（仅将 ~/Codes/d3LLM 换为仓库根目录 REPO_ROOT）。
# 用法:
#   bash eval_scripts/run_code_eval.sh <recipe> [MODEL]
#   MODEL 可选: 覆盖该条 recipe 中的 HF id / 本地路径（pretrained= / model_path= / CKPT_DIR / --model）
#   MERGED_MODEL_PATH: 仅 recipe dream_gsm8k_cot__merged_mblock_kv_delay1 在未传 MODEL 时使用（默认 ${REPO_ROOT}/output_model/merged_d3LLM_DREAM_5742，与 dream_gsm8k_cot.sh 一致）
#
#   bash eval_scripts/run_code_eval.sh list
#   EVALPLUS_DEBUG=1: 打印 dream_coder 相关路径与 python -P 分支（排查 evalplus 导入）
#
# 对照: 各 recipe 名 ≈ eval_scripts/<原脚本> 中对应注释块；详见函数上方注释。
#
set -euo pipefail

# REPO_ROOT：必须基于「本脚本」的绝对路径解析。若仅用 cd "$(dirname "${BASH_SOURCE[0]}")/.."，
# 在 BASH_SOURCE 为相对路径时，会相对于当前工作目录 cd，cwd 不在仓库根时会得到错误的 REPO_ROOT，
# 进而 EVALPLUS_ROOT 指向不存在的位置，出现 No module named evalplus.evaluate。
_script_path="${BASH_SOURCE[0]}"
[[ "${_script_path}" != /* ]] && _script_path="$(pwd)/${_script_path}"
SCRIPT_DIR="$(cd "$(dirname "${_script_path}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
unset _script_path

MODEL_OVERRIDE="${2:-}"
# dream-coder / evalplus：原脚本写 PYTHONPATH=evalplus（相对 code_eval）。用绝对路径指向 vendored evalplus。
EVALPLUS_ROOT="${REPO_ROOT}/utils/utils_DreamCoder/code_eval/evalplus"
if [[ ! -f "${EVALPLUS_ROOT}/evalplus/evaluate.py" ]]; then
    echo "错误: 未找到 EvalPlus 模块，期望路径: ${EVALPLUS_ROOT}/evalplus/evaluate.py（REPO_ROOT=${REPO_ROOT}）" >&2
    exit 1
fi

# 在 code_eval 目录下跑 python 时，sys.path 会把 cwd 放最前，优先加载外层 evalplus/__init__.py（仓库根的空包），
# 真正的 evaluate 在内层 evalplus/evaluate.py，于是报错 No module named evalplus.evaluate。
# Python 3.11+ 的 -P 禁止把 cwd 加入 sys.path，可消除该冲突；旧版本则改为在 REPO_ROOT 下执行。
_evalplus_run() {
    export PYTHONPATH="${EVALPLUS_ROOT}"
    if [[ "${EVALPLUS_DEBUG:-0}" == "1" ]]; then
        echo "[run_code_eval][evalplus] cwd=$(pwd)" >&2
        echo "[run_code_eval][evalplus] REPO_ROOT=${REPO_ROOT}" >&2
        echo "[run_code_eval][evalplus] EVALPLUS_ROOT=${EVALPLUS_ROOT}" >&2
        echo "[run_code_eval][evalplus] PYTHONPATH=${PYTHONPATH}" >&2
        echo "[run_code_eval][evalplus] python=$(command -v python) $("${PYTHON:-python}" --version 2>&1)" >&2
    fi
    if "${PYTHON:-python}" -P -c "pass" 2>/dev/null; then
        [[ "${EVALPLUS_DEBUG:-0}" == "1" ]] && echo "[run_code_eval][evalplus] 使用 python -P（避免 cwd 覆盖 evalplus 包）" >&2
        "${PYTHON:-python}" -P -m evalplus.evaluate "$@"
        return
    fi
    [[ "${EVALPLUS_DEBUG:-0}" == "1" ]] && echo "[run_code_eval][evalplus] python 无 -P，改为在 REPO_ROOT 下执行" >&2
    (
        cd "${REPO_ROOT}" || exit 1
        export PYTHONPATH="${EVALPLUS_ROOT}"
        "${PYTHON:-python}" -m evalplus.evaluate "$@"
    )
}

usage() {
    sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
    echo "可用 recipe 见: bash $0 list"
}

# ---------- dream_gsm8k_cot.sh: Vanilla Dream TPF=1.0 ----------
dream_gsm8k_cot__vanilla_entropy() {
    cd "${REPO_ROOT}/utils/utils_Dream/eval_instruct" || exit 1
    local P="${MODEL_OVERRIDE:-Dream-org/Dream-v0-Instruct-7B}"
    CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
        --model diffllm \
        --model_args torch_compile=False,pretrained="${P}",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
        --tasks gsm8k_cot_zeroshot \
        --device cuda \
        --batch_size 1 \
        --num_fewshot 0 \
        --output_path eval_tmp/gsm8k_cot_zeroshot \
        --log_samples --confirm_run_unsafe_code \
        --apply_chat_template
}

# dream_gsm8k_cot.sh: 合并权重示例（原脚本 L71-73，pretrained 为本地目录）
dream_gsm8k_cot__merged_mblock_kv_delay1() {
    cd "${REPO_ROOT}/utils/utils_Dream/eval_instruct" || exit 1
    local P="${MODEL_OVERRIDE:-${MERGED_MODEL_PATH:-${REPO_ROOT}/output_model/merged_d3LLM_DREAM_5742}}"
    accelerate launch --main_process_port 46666 -m lm_eval --model diffllm --model_args torch_compile=False,pretrained="${P}",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.,alg=entropy_threshold,dParallel=False,threshold=0.4,generation_method=generation_multi_block,block_add_threshold=0.1,decoded_token_threshold=0.95,block_length=32,cache_delay_iter=1,refresh_interval=10000,early_stop=True --tasks gsm8k_cot_zeroshot --device cuda --batch_size 1 --num_fewshot 0 --output_path ./eval_tmp/multi_block_cot --log_samples --confirm_run_unsafe_code --apply_chat_template
}

dream_gsm8k_cot__d3llm_entropy() {
    cd "${REPO_ROOT}/utils/utils_Dream/eval_instruct" || exit 1
    local P="${MODEL_OVERRIDE:-d3LLM/d3LLM_Dream}"
    accelerate launch --main_process_port 46666 -m lm_eval --model diffllm --model_args pretrained="${P}",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.1,top_p=0.9,alg=entropy,dParallel=False --tasks gsm8k_cot_zeroshot --device cuda --batch_size 1 --num_fewshot 0 --output_path ./eval_tmp/multi_block_cot --log_samples --confirm_run_unsafe_code --apply_chat_template
}

dream_gsm8k_cot__d3llm_multiblock_nodelay() {
    cd "${REPO_ROOT}/utils/utils_Dream/eval_instruct" || exit 1
    local P="${MODEL_OVERRIDE:-d3LLM/d3LLM_Dream}"
    accelerate launch --main_process_port 46666 -m lm_eval --model diffllm --model_args torch_compile=True,pretrained="${P}",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.,alg=entropy_threshold,dParallel=False,threshold=0.4,generation_method=generation_multi_block,block_add_threshold=0.1,decoded_token_threshold=0.95,block_length=32,cache_delay_iter=10000 --tasks gsm8k_cot_zeroshot --device cuda --batch_size 1 --num_fewshot 0 --output_path ./eval_tmp/multi_block_cot --log_samples --confirm_run_unsafe_code --apply_chat_template
}

dream_gsm8k_cot__d3llm_multiblock_kv_delay1() {
    cd "${REPO_ROOT}/utils/utils_Dream/eval_instruct" || exit 1
    local P="${MODEL_OVERRIDE:-d3LLM/d3LLM_Dream}"
    accelerate launch --main_process_port 46666 -m lm_eval --model diffllm --model_args torch_compile=False,pretrained="${P}",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.,alg=entropy_threshold,dParallel=False,threshold=0.4,generation_method=generation_multi_block,block_add_threshold=0.1,decoded_token_threshold=0.95,block_length=32,cache_delay_iter=1,refresh_interval=10000,early_stop=True --tasks gsm8k_cot_zeroshot --device cuda --batch_size 1 --num_fewshot 0 --output_path ./eval_tmp/multi_block_cot --log_samples --confirm_run_unsafe_code --apply_chat_template
}

# ---------- dream_humaneval.sh ----------
dream_humaneval__vanilla_entropy() {
    cd "${REPO_ROOT}/utils/utils_Dream/eval_instruct" || exit 1
    export HF_ALLOW_CODE_EVAL=1
    local P="${MODEL_OVERRIDE:-Dream-org/Dream-v0-Instruct-7B}"
    PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval --model diffllm --model_args torch_compile=False,pretrained="${P}",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.1,top_p=0.9,alg=entropy,dParallel=False --tasks humaneval_instruct --device cuda --batch_size 1 --num_fewshot 0 --output_path ./eval_tmp/multi_block_cot --log_samples --confirm_run_unsafe_code --apply_chat_template
}

dream_humaneval__d3llm_entropy() {
    export HF_ALLOW_CODE_EVAL=1
    cd "${REPO_ROOT}/utils/utils_Dream/eval_instruct" || exit 1
    local P="${MODEL_OVERRIDE:-d3LLM/d3LLM_Dream}"
    PYTHONPATH=. accelerate launch --main_process_port 46666 -m lm_eval --model diffllm --model_args pretrained="${P}",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.1,top_p=0.9,alg=entropy,dParallel=False --tasks humaneval_instruct --device cuda --batch_size 1 --num_fewshot 0 --output_path ./eval_tmp/multi_block_cot --log_samples --confirm_run_unsafe_code --apply_chat_template
}

dream_humaneval__d3llm_multiblock_t01() {
    export HF_ALLOW_CODE_EVAL=1
    cd "${REPO_ROOT}/utils/utils_Dream/eval_instruct" || exit 1
    local P="${MODEL_OVERRIDE:-d3LLM/d3LLM_Dream}"
    PYTHONPATH=. accelerate launch --main_process_port 46666 -m lm_eval --model diffllm --model_args torch_compile=True,pretrained="${P}",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.1,alg=entropy_threshold,dParallel=False,threshold=0.5,generation_method=generation_multi_block,block_add_threshold=0.1,decoded_token_threshold=0.95,block_length=32,cache_delay_iter=10000,early_stop=True --tasks humaneval_instruct --device cuda --batch_size 1 --num_fewshot 0 --output_path ./eval_tmp/multi_block_cot --log_samples --confirm_run_unsafe_code --apply_chat_template
}

dream_humaneval__d3llm_multiblock_delay2() {
    export HF_ALLOW_CODE_EVAL=1
    cd "${REPO_ROOT}/utils/utils_Dream/eval_instruct" || exit 1
    local P="${MODEL_OVERRIDE:-d3LLM/d3LLM_Dream}"
    PYTHONPATH=. accelerate launch --main_process_port 46666 -m lm_eval --model diffllm --model_args torch_compile=False,pretrained="${P}",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.1,alg=entropy_threshold,dParallel=False,threshold=0.5,generation_method=generation_multi_block,block_add_threshold=0.1,decoded_token_threshold=0.95,block_length=32,cache_delay_iter=2,early_stop=True,refresh_interval=10000 --tasks humaneval_instruct --device cuda --batch_size 1 --num_fewshot 0 --output_path ./eval_tmp/multi_block_cot --log_samples --confirm_run_unsafe_code --apply_chat_template
}

# ---------- dream_long_gsm8k.sh ----------
dream_long_gsm8k__vanilla_entropy() {
    cd "${REPO_ROOT}/utils/utils_Dream/eval_instruct" || exit 1
    local P="${MODEL_OVERRIDE:-Dream-org/Dream-v0-Instruct-7B}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
        --model diffllm \
        --model_args torch_compile=False,pretrained="${P}",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
        --tasks gsm8k \
        --device cuda \
        --batch_size 1 \
        --num_fewshot 5 \
        --output_path eval_tmp/gsm8k \
        --log_samples --confirm_run_unsafe_code \
        --apply_chat_template
}

dream_long_gsm8k__d3llm_multiblock_nodelay() {
    cd "${REPO_ROOT}/utils/utils_Dream/eval_instruct" || exit 1
    local P="${MODEL_OVERRIDE:-d3LLM/d3LLM_Dream}"
    accelerate launch --main_process_port 46666 -m lm_eval --model diffllm --model_args torch_compile=True,pretrained="${P}",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.1,alg=entropy_threshold,dParallel=False,threshold=0.4,generation_method=generation_multi_block,block_add_threshold=0.1,decoded_token_threshold=0.95,block_length=32,cache_delay_iter=10000,refresh_interval=10000 --tasks gsm8k --device cuda --batch_size 1 --num_fewshot 5 --output_path ./eval_tmp/multi_block_cot --log_samples --confirm_run_unsafe_code --apply_chat_template
}

dream_long_gsm8k__d3llm_multiblock_kv_delay1() {
    cd "${REPO_ROOT}/utils/utils_Dream/eval_instruct" || exit 1
    local P="${MODEL_OVERRIDE:-d3LLM/d3LLM_Dream}"
    accelerate launch --main_process_port 46666 -m lm_eval --model diffllm --model_args torch_compile=False,pretrained="${P}",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.,alg=entropy_threshold,dParallel=False,threshold=0.45,generation_method=generation_multi_block,block_add_threshold=0.1,decoded_token_threshold=0.95,block_length=32,cache_delay_iter=1,refresh_interval=10000,early_stop=True --tasks gsm8k --device cuda --batch_size 1 --num_fewshot 5 --output_path ./eval_tmp/multi_block_cot --log_samples --confirm_run_unsafe_code --apply_chat_template
}

# ---------- dream_math.sh ----------
dream_math__d3llm_multiblock_nodelay() {
    cd "${REPO_ROOT}/utils/utils_Dream/eval_instruct" || exit 1
    local P="${MODEL_OVERRIDE:-d3LLM/d3LLM_Dream}"
    accelerate launch --main_process_port 12334 -m lm_eval --model diffllm --model_args pretrained="${P}",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.,alg=entropy_threshold,dParallel=False,threshold=0.4,generation_method=generation_multi_block,block_add_threshold=0.1,decoded_token_threshold=0.95,block_length=32,cache_delay_iter=10000 --tasks minerva_math --device cuda --batch_size 1 --num_fewshot 0 --output_path ./eval_tmp/minerva_4 --log_samples --confirm_run_unsafe_code --apply_chat_template
}

dream_math__d3llm_multiblock_delay2() {
    cd "${REPO_ROOT}/utils/utils_Dream/eval_instruct" || exit 1
    local P="${MODEL_OVERRIDE:-d3LLM/d3LLM_Dream}"
    accelerate launch --main_process_port 12334 -m lm_eval --model diffllm --model_args torch_compile=False,pretrained="${P}",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.,alg=entropy_threshold,dParallel=False,threshold=0.4,generation_method=generation_multi_block,block_add_threshold=0.1,decoded_token_threshold=0.95,block_length=32,cache_delay_iter=2,refresh_interval=10000,early_stop=True --tasks minerva_math --device cuda --batch_size 1 --num_fewshot 0 --output_path ./eval_tmp/minerva_4 --log_samples --confirm_run_unsafe_code --apply_chat_template
}

# ---------- dream_mbpp.sh ----------
dream_mbpp__d3llm_multiblock_nodelay() {
    export HF_ALLOW_CODE_EVAL=1
    cd "${REPO_ROOT}/utils/utils_Dream/eval_instruct" || exit 1
    local P="${MODEL_OVERRIDE:-d3LLM/d3LLM_Dream}"
    accelerate launch --main_process_port 46666 -m lm_eval --model diffllm --model_args torch_compile=True,pretrained="${P}",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.,alg=entropy_threshold,dParallel=False,threshold=0.4,generation_method=generation_multi_block,block_add_threshold=0.1,decoded_token_threshold=0.95,block_length=32,cache_delay_iter=10000 --tasks mbpp_instruct --device cuda --batch_size 1 --num_fewshot 4 --output_path ./eval_tmp/multi_block_cot --log_samples --confirm_run_unsafe_code --apply_chat_template
}

dream_mbpp__d3llm_entropy() {
    cd "${REPO_ROOT}/utils/utils_Dream/eval_instruct" || exit 1
    local P="${MODEL_OVERRIDE:-d3LLM/d3LLM_Dream}"
    accelerate launch --main_process_port 46666 -m lm_eval --model diffllm --model_args pretrained="${P}",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.1,top_p=0.9,alg=entropy,dParallel=False --tasks mbpp_instruct --device cuda --batch_size 1 --num_fewshot 4 --output_path ./eval_tmp/multi_block_cot --log_samples --confirm_run_unsafe_code --apply_chat_template
}

dream_mbpp__d3llm_multiblock_delay2() {
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
    export HF_ALLOW_CODE_EVAL=1
    cd "${REPO_ROOT}/utils/utils_Dream/eval_instruct" || exit 1
    local P="${MODEL_OVERRIDE:-d3LLM/d3LLM_Dream}"
    accelerate launch --main_process_port 46666 -m lm_eval --model diffllm --model_args torch_compile=False,pretrained="${P}",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.,alg=entropy_threshold,dParallel=False,threshold=0.45,generation_method=generation_multi_block,block_add_threshold=0.1,decoded_token_threshold=0.95,block_length=32,cache_delay_iter=2,refresh_interval=10000,early_stop=True --tasks mbpp_instruct --device cuda --batch_size 1 --num_fewshot 4 --output_path ./eval_tmp/multi_block_cot --log_samples --confirm_run_unsafe_code --apply_chat_template
}

# ---------- dream-coder.sh ----------
dream_coder__qwen_hf_humaneval() {
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
    cd "${REPO_ROOT}/utils/utils_DreamCoder/code_eval" || exit 1
    CKPT_DIR="${MODEL_OVERRIDE:-Qwen/Qwen2.5-Coder-7B-Instruct}"
    _evalplus_run --model "${CKPT_DIR}" --trust_remote_code True --max_new_tokens 512 --diffusion_steps 512 --dataset humaneval --backend hf --temperature 0.1
}

dream_coder__qwen_hf_mbpp() {
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
    cd "${REPO_ROOT}/utils/utils_DreamCoder/code_eval" || exit 1
    CKPT_DIR="${MODEL_OVERRIDE:-Qwen/Qwen2.5-Coder-7B-Instruct}"
    _evalplus_run --model "${CKPT_DIR}" --trust_remote_code True --max_new_tokens 512 --diffusion_steps 512 --dataset mbpp --backend hf --temperature 0.1
}

dream_coder__humaneval_vanilla_dllm() {
    cd "${REPO_ROOT}/utils/utils_DreamCoder/code_eval" || exit 1
    CKPT_DIR="${MODEL_OVERRIDE:-Dream-org/Dream-Coder-v0-Instruct-7B}"
    _evalplus_run --model "${CKPT_DIR}" --trust_remote_code True --max_new_tokens 512 --diffusion_steps 512 --dataset humaneval --backend dllm --temperature 0.1
}

dream_coder__humaneval_d3llm_multiblock() {
    cd "${REPO_ROOT}/utils/utils_DreamCoder/code_eval" || exit 1
    CKPT_DIR="${MODEL_OVERRIDE:-d3LLM/d3LLM_Dream_Coder}"
    _evalplus_run --model "${CKPT_DIR}" --trust_remote_code True --max_new_tokens 512 --diffusion_steps 512 --dataset humaneval --backend dllm --temperature 0. --generation_method generation_multi_block --alg entropy_threshold --threshold 0.5 --block_length 32 --block_add_threshold 0.1 --decoded_token_threshold 0.95 --cache_delay_iter 32 --early_stop True --torch_compile True
}

dream_coder__mbpp_vanilla_dllm() {
    cd "${REPO_ROOT}/utils/utils_DreamCoder/code_eval" || exit 1
    CKPT_DIR="${MODEL_OVERRIDE:-Dream-org/Dream-Coder-v0-Instruct-7B}"
    _evalplus_run --model "${CKPT_DIR}" --trust_remote_code True --max_new_tokens 512 --diffusion_steps 512 --dataset mbpp --backend dllm --temperature 0.1
}

dream_coder__mbpp_d3llm_multiblock() {
    cd "${REPO_ROOT}/utils/utils_DreamCoder/code_eval" || exit 1
    CKPT_DIR="${MODEL_OVERRIDE:-d3LLM/d3LLM_Dream_Coder}"
    _evalplus_run --model "${CKPT_DIR}" --trust_remote_code True --max_new_tokens 512 --diffusion_steps 512 --dataset mbpp --backend dllm --temperature 0. --generation_method generation_multi_block --alg entropy_threshold --threshold 0.5 --block_length 32 --block_add_threshold 0.1 --decoded_token_threshold 0.95 --cache_delay_iter 32 --early_stop True --torch_compile True
}

# ---------- llada_gsm8k_cot.sh ----------
# L82-86 d3LLM-LLaDA TPF=1.0（无 HF_ALLOW）
llada_gsm8k_cot__d3llm_tpf() {
    cd "${REPO_ROOT}/utils/utils_LLaDA" || exit 1
    local MP="${MODEL_OVERRIDE:-d3LLM/d3LLM_LLaDA}"
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29600 eval_llada.py --tasks gsm8k_cot_zeroshot --num_fewshot 0 \
        --confirm_run_unsafe_code --model llada_dist \
        --model_args model_path="${MP}",gen_length=256,steps=256,block_length=32,show_speed=True,task="gsm8k_cot_zeroshot" --batch_size 1
}

# L89-100 generate_multi_block
llada_gsm8k_cot__d3llm_generate_multi_block() {
    cd "${REPO_ROOT}/utils/utils_LLaDA" || exit 1
    export HF_ALLOW_CODE_EVAL=1
    export HF_DATASETS_TRUST_REMOTE_CODE=true
    task=gsm8k_cot_zeroshot
    length=256
    block_length=32
    num_fewshot=0
    steps=256
    local MP="${MODEL_OVERRIDE:-d3LLM/d3LLM_LLaDA}"
    accelerate launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
        --confirm_run_unsafe_code --model llada_dist \
        --model_args model_path="${MP}",gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.5,task="gsm8k_cot_zeroshot",generation_method="generate_multi_block",early_stop=True
}

# L103-114 generate_multi_block_kv_cache delay=2
llada_gsm8k_cot__d3llm_kv_delay2() {
    cd "${REPO_ROOT}/utils/utils_LLaDA" || exit 1
    export HF_ALLOW_CODE_EVAL=1
    export HF_DATASETS_TRUST_REMOTE_CODE=true
    task=gsm8k_cot_zeroshot
    length=256
    block_length=32
    num_fewshot=0
    steps=256
    local MP="${MODEL_OVERRIDE:-d3LLM/d3LLM_LLaDA}"
    accelerate launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
        --confirm_run_unsafe_code --model llada_dist \
        --model_args model_path="${MP}",gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.5,task="gsm8k_cot_zeroshot",generation_method="generate_multi_block_kv_cache",cache_delay_iter=2,refresh_interval=3,early_stop=True
}

# ---------- llada_humaneval.sh d3LLM 块 ----------
llada_humaneval__d3llm_tpf() {
    cd "${REPO_ROOT}/utils/utils_LLaDA" || exit 1
    export HF_ALLOW_CODE_EVAL=1
    export HF_DATASETS_TRUST_REMOTE_CODE=true
    task=humaneval
    length=256
    block_length=32
    num_fewshot=0
    steps=256
    model_path="${MODEL_OVERRIDE:-d3LLM/d3LLM_LLaDA}"
    output_dir='d3llm'
    METHOD_NAME_ENCODED=$(echo "${model_path}" | sed 's|/|__|g')
    accelerate launch --main_process_port 29600 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
        --confirm_run_unsafe_code --model llada_dist \
        --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,task="${task}" \
        --output_path evals_results/${output_dir}/${task}-ns${num_fewshot}-${length} --log_samples
    latest_jsonl=$(find evals_results/${output_dir}/${task}-ns${num_fewshot}-${length}/${METHOD_NAME_ENCODED} -name "samples_${task}_*.jsonl" -type f 2>/dev/null | head -n 1)
    [ -n "$latest_jsonl" ] && python postprocess_code_humaneval.py "$latest_jsonl" || echo "No jsonl file found"
}

llada_humaneval__d3llm_entropy_multiblock() {
    cd "${REPO_ROOT}/utils/utils_LLaDA" || exit 1
    export HF_ALLOW_CODE_EVAL=1
    export HF_DATASETS_TRUST_REMOTE_CODE=true
    task=humaneval
    length=256
    block_length=32
    num_fewshot=0
    steps=256
    model_path="${MODEL_OVERRIDE:-d3LLM/d3LLM_LLaDA}"
    output_dir='d3llm'
    METHOD_NAME_ENCODED=$(echo "${model_path}" | sed 's|/|__|g')
    accelerate launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
        --confirm_run_unsafe_code --model llada_dist \
        --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.5,task="${task}" \
        --output_path evals_results/${output_dir}/${task}-entropy-ns${num_fewshot}-${length} --log_samples
    latest_jsonl=$(find evals_results/${output_dir}/${task}-entropy-ns${num_fewshot}-${length}/${METHOD_NAME_ENCODED} -name "samples_${task}_*.jsonl" -type f 2>/dev/null | head -n 1)
    [ -n "$latest_jsonl" ] && python postprocess_code_humaneval.py "$latest_jsonl" || echo "No jsonl file found"
}

llada_humaneval__d3llm_kv_delay2() {
    cd "${REPO_ROOT}/utils/utils_LLaDA" || exit 1
    export HF_ALLOW_CODE_EVAL=1
    export HF_DATASETS_TRUST_REMOTE_CODE=true
    task=humaneval
    length=256
    block_length=32
    num_fewshot=0
    steps=256
    model_path="${MODEL_OVERRIDE:-d3LLM/d3LLM_LLaDA}"
    output_dir='d3llm'
    METHOD_NAME_ENCODED=$(echo "${model_path}" | sed 's|/|__|g')
    accelerate launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
        --confirm_run_unsafe_code --model llada_dist \
        --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.5,task="${task}",generation_method="generate_multi_block_kv_cache",cache_delay_iter=2,refresh_interval=4,block_add_threshold=1.0,decoded_token_threshold=1.0,block_length=32,early_stop=True \
        --output_path evals_results/${output_dir}/${task}-multi-block-ns${num_fewshot}-${length} --log_samples
    latest_jsonl=$(find evals_results/${output_dir}/${task}-multi-block-ns${num_fewshot}-${length}/${METHOD_NAME_ENCODED} -name "samples_${task}_*.jsonl" -type f 2>/dev/null | head -n 1)
    [ -n "$latest_jsonl" ] && python postprocess_code_humaneval.py "$latest_jsonl" || echo "No jsonl file found"
}

# ---------- llada_mbpp.sh d3LLM ----------
llada_mbpp__d3llm_generate_multi_block() {
    cd "${REPO_ROOT}/utils/utils_LLaDA" || exit 1
    export HF_ALLOW_CODE_EVAL=1
    export HF_DATASETS_TRUST_REMOTE_CODE=true
    task=mbpp
    length=256
    block_length=32
    num_fewshot=3
    steps=256
    model_path="${MODEL_OVERRIDE:-d3LLM/d3LLM_LLaDA}"
    output_dir='d3llm'
    METHOD_NAME_ENCODED=$(echo "${model_path}" | sed 's|/|__|g')
    accelerate launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
        --confirm_run_unsafe_code --model llada_dist \
        --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.4,task="${task}",generation_method="generate_multi_block",block_add_threshold=0.1,decoded_token_threshold=0.95,block_length=32,early_stop=True \
        --output_path evals_results/${output_dir}/${task}-multi-block-ns${num_fewshot}-${length} --log_samples
    latest_jsonl=$(find evals_results/${output_dir}/${task}-multi-block-ns${num_fewshot}-${length}/${METHOD_NAME_ENCODED} -name "samples_${task}_*.jsonl" -type f 2>/dev/null | head -n 1)
    [ -n "$latest_jsonl" ] && python postprocess_code_mbpp.py "$latest_jsonl" || echo "No jsonl file found"
}

llada_mbpp__d3llm_kv_delay1() {
    cd "${REPO_ROOT}/utils/utils_LLaDA" || exit 1
    export HF_ALLOW_CODE_EVAL=1
    export HF_DATASETS_TRUST_REMOTE_CODE=true
    task=mbpp
    length=256
    block_length=32
    num_fewshot=3
    steps=256
    model_path="${MODEL_OVERRIDE:-d3LLM/d3LLM_LLaDA}"
    output_dir='d3llm'
    METHOD_NAME_ENCODED=$(echo "${model_path}" | sed 's|/|__|g')
    accelerate launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
        --confirm_run_unsafe_code --model llada_dist \
        --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.4,task="${task}",generation_method="generate_multi_block_kv_cache",cache_delay_iter=1,refresh_interval=10000,block_add_threshold=0.1,decoded_token_threshold=0.95,block_length=32,early_stop=True \
        --output_path evals_results/${output_dir}/${task}-multi-block-ns${num_fewshot}-${length} --log_samples
    latest_jsonl=$(find evals_results/${output_dir}/${task}-multi-block-ns${num_fewshot}-${length}/${METHOD_NAME_ENCODED} -name "samples_${task}_*.jsonl" -type f 2>/dev/null | head -n 1)
    [ -n "$latest_jsonl" ] && python postprocess_code_mbpp.py "$latest_jsonl" || echo "No jsonl file found"
}

# ---------- llada_math.sh d3LLM ----------
llada_math__d3llm_tpf() {
    cd "${REPO_ROOT}/utils/utils_LLaDA" || exit 1
    local MP="${MODEL_OVERRIDE:-d3LLM/d3LLM_LLaDA}"
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29600 eval_llada.py --tasks minerva_math --num_fewshot 4 \
        --confirm_run_unsafe_code --model llada_dist \
        --model_args model_path="${MP}",gen_length=256,steps=256,block_length=32,show_speed=True,task="minerva_math" --batch_size 1
}

llada_math__d3llm_generate_multi_block() {
    cd "${REPO_ROOT}/utils/utils_LLaDA" || exit 1
    export HF_ALLOW_CODE_EVAL=1
    export HF_DATASETS_TRUST_REMOTE_CODE=true
    task=minerva_math
    length=256
    block_length=32
    num_fewshot=4
    steps=256
    local MP="${MODEL_OVERRIDE:-d3LLM/d3LLM_LLaDA}"
    accelerate launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
        --confirm_run_unsafe_code --model llada_dist \
        --model_args model_path="${MP}",gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.5,task="minerva_math",generation_method="generate_multi_block",block_add_threshold=0.1,decoded_token_threshold=0.95,block_length=32,early_stop=True
}

llada_math__d3llm_kv_delay1() {
    cd "${REPO_ROOT}/utils/utils_LLaDA" || exit 1
    export HF_ALLOW_CODE_EVAL=1
    export HF_DATASETS_TRUST_REMOTE_CODE=true
    task=minerva_math
    length=256
    block_length=32
    num_fewshot=4
    steps=256
    local MP="${MODEL_OVERRIDE:-d3LLM/d3LLM_LLaDA}"
    accelerate launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
        --confirm_run_unsafe_code --model llada_dist \
        --model_args model_path="${MP}",gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.5,task="minerva_math",generation_method="generate_multi_block_kv_cache",cache_delay_iter=1,refresh_interval=10000,block_add_threshold=0.1,decoded_token_threshold=0.95,block_length=32,early_stop=True
}

list_recipes() {
    grep -E '^[a-z0-9_]+\(\) \{$' "$0" | sed 's/() {$//' | grep -v '^list_recipes$' | grep -v '^usage$'
}

case "${1:-}" in
    ""|-h|--help|help)
        usage
        exit 0
        ;;
    list)
        list_recipes
        exit 0
        ;;
    dream_gsm8k_cot__vanilla_entropy) dream_gsm8k_cot__vanilla_entropy ;;
    dream_gsm8k_cot__merged_mblock_kv_delay1) dream_gsm8k_cot__merged_mblock_kv_delay1 ;;
    dream_gsm8k_cot__d3llm_entropy) dream_gsm8k_cot__d3llm_entropy ;;
    dream_gsm8k_cot__d3llm_multiblock_nodelay) dream_gsm8k_cot__d3llm_multiblock_nodelay ;;
    dream_gsm8k_cot__d3llm_multiblock_kv_delay1) dream_gsm8k_cot__d3llm_multiblock_kv_delay1 ;;
    dream_humaneval__vanilla_entropy) dream_humaneval__vanilla_entropy ;;
    dream_humaneval__d3llm_entropy) dream_humaneval__d3llm_entropy ;;
    dream_humaneval__d3llm_multiblock_t01) dream_humaneval__d3llm_multiblock_t01 ;;
    dream_humaneval__d3llm_multiblock_delay2) dream_humaneval__d3llm_multiblock_delay2 ;;
    dream_long_gsm8k__vanilla_entropy) dream_long_gsm8k__vanilla_entropy ;;
    dream_long_gsm8k__d3llm_multiblock_nodelay) dream_long_gsm8k__d3llm_multiblock_nodelay ;;
    dream_long_gsm8k__d3llm_multiblock_kv_delay1) dream_long_gsm8k__d3llm_multiblock_kv_delay1 ;;
    dream_math__d3llm_multiblock_nodelay) dream_math__d3llm_multiblock_nodelay ;;
    dream_math__d3llm_multiblock_delay2) dream_math__d3llm_multiblock_delay2 ;;
    dream_mbpp__d3llm_multiblock_nodelay) dream_mbpp__d3llm_multiblock_nodelay ;;
    dream_mbpp__d3llm_entropy) dream_mbpp__d3llm_entropy ;;
    dream_mbpp__d3llm_multiblock_delay2) dream_mbpp__d3llm_multiblock_delay2 ;;
    dream_coder__qwen_hf_humaneval) dream_coder__qwen_hf_humaneval ;;
    dream_coder__qwen_hf_mbpp) dream_coder__qwen_hf_mbpp ;;
    dream_coder__humaneval_vanilla_dllm) dream_coder__humaneval_vanilla_dllm ;;
    dream_coder__humaneval_d3llm_multiblock) dream_coder__humaneval_d3llm_multiblock ;;
    dream_coder__mbpp_vanilla_dllm) dream_coder__mbpp_vanilla_dllm ;;
    dream_coder__mbpp_d3llm_multiblock) dream_coder__mbpp_d3llm_multiblock ;;
    llada_gsm8k_cot__d3llm_tpf) llada_gsm8k_cot__d3llm_tpf ;;
    llada_gsm8k_cot__d3llm_generate_multi_block) llada_gsm8k_cot__d3llm_generate_multi_block ;;
    llada_gsm8k_cot__d3llm_kv_delay2) llada_gsm8k_cot__d3llm_kv_delay2 ;;
    llada_humaneval__d3llm_tpf) llada_humaneval__d3llm_tpf ;;
    llada_humaneval__d3llm_entropy_multiblock) llada_humaneval__d3llm_entropy_multiblock ;;
    llada_humaneval__d3llm_kv_delay2) llada_humaneval__d3llm_kv_delay2 ;;
    llada_mbpp__d3llm_generate_multi_block) llada_mbpp__d3llm_generate_multi_block ;;
    llada_mbpp__d3llm_kv_delay1) llada_mbpp__d3llm_kv_delay1 ;;
    llada_math__d3llm_tpf) llada_math__d3llm_tpf ;;
    llada_math__d3llm_generate_multi_block) llada_math__d3llm_generate_multi_block ;;
    llada_math__d3llm_kv_delay1) llada_math__d3llm_kv_delay1 ;;
    *)
        echo "未知 recipe: $1" >&2
        echo "运行: bash $0 list" >&2
        exit 1
        ;;
esac
