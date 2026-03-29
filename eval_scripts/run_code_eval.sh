#!/usr/bin/env bash
# 代码 / 通用 lm_eval 评测入口（Dream diffllm、evalplus、LLaDA 预设）
#
# ========== 1) diffllm：模型 + 任务 + [m] [s] [t] ==========
#   bash eval_scripts/run_code_eval.sh diffllm <模型路径或HF仓库id> <task> [m] [s] [t]
#
#   后端 DIFFLLM_BACKEND（默认 auto）:
#     auto — humaneval_instruct / mbpp_instruct 走 evalplus.evaluate（与 dream-coder.sh 第16–20行
#            同款 multi_block + dllm）；其它任务仍走 lm_eval + diffllm。
#     lm_eval — 强制 lm_eval（代码任务也用 instruct 任务与 lm_eval 指标）。
#     evalplus — 强制 evalplus（仅支持上述代码类任务名）。
#
#   与 dream-coder.sh 一致时无需改命令；若要坚持 lm_eval 的 humaneval_instruct：
#     DIFFLLM_BACKEND=lm_eval bash eval_scripts/run_code_eval.sh diffllm ...
#
#   模型解析：若 <模型> 为已存在目录且含 config.json（绝对路径、当前目录相对路径、或相对仓库根
#   的路径），则按本地合并模型使用；否则视为 Hugging Face 上的 repo id（如 d3LLM/d3LLM_Dream_Coder）。
#
#   可选环境变量: NUM_FEWSHOT（mbpp_instruct 未设置时默认 4）、EVAL_OUTPUT_DIR
#   多卡: 若已设置 CUDA_VISIBLE_DEVICES=0,1,...（多张卡）且未设置 ACCELERATE_NUM_PROCESSES，
#   则自动令进程数=可见 GPU 数（数据并行）；若只想单卡评测请设 ACCELERATE_NUM_PROCESSES=1
#
#   任务简写: humaneval -> humaneval_instruct, mbpp -> mbpp_instruct
#
#   示例:
#     bash eval_scripts/run_code_eval.sh diffllm d3LLM/d3LLM_Dream_Coder humaneval_instruct 256 256 0.4
#     bash eval_scripts/run_code_eval.sh diffllm output_model/merged_d3LLM_DREAM_Coder humaneval 256 256 0.4
#
# ========== 2) evalplus：HumanEval / MBPP（dllm 后端）==========
#   bash eval_scripts/run_code_eval.sh evalplus <模型路径或HF id> <humaneval|mbpp> [multi_block|vanilla]
#
# ========== 3) 旧版预设 ==========
#   bash eval_scripts/run_code_eval.sh <模型> <数据集>
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D3LLM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    cat <<'EOF'
用法摘要:

  【diffllm】模型 + 任务 + [m] [s] [t]
    humaneval_instruct / mbpp_instruct 默认走 evalplus（与 dream-coder.sh 一致）；
    其它任务走 lm_eval。强制 lm_eval: DIFFLLM_BACKEND=lm_eval

  【evalplus】
    bash eval_scripts/run_code_eval.sh evalplus <模型> <humaneval|mbpp> [multi_block|vanilla]

  【旧版预设】
    export HF_ALLOW_CODE_EVAL=1
    bash eval_scripts/run_code_eval.sh <模型> <数据集>

详见脚本顶部注释。
EOF
}

# ---------- 解析模型：本地（config.json）或 Hugging Face id ----------
resolve_pretrained() {
    local raw="${1:?模型参数不能为空}"
    local under="${D3LLM_ROOT}/${raw}"

    if [[ -d "$raw" ]]; then
        if [[ ! -f "${raw}/config.json" ]]; then
            echo "错误: 本地目录存在但缺少 config.json: $raw" >&2
            return 1
        fi
        (cd "$raw" && pwd)
        return 0
    fi
    if [[ -d "$under" ]] && [[ -f "${under}/config.json" ]]; then
        echo "$under"
        return 0
    fi
    echo "$raw"
    return 0
}

# 与 eval_scripts/dream-coder.sh 中 d3LLM Dream-Coder multi_block 段一致（须在 code_eval/evalplus 目录下执行）
_dream_coder_evalplus_multi_block() {
    local ckpt="$1"
    local dataset="$2"
    local mnt="$3"
    local dst="$4"
    local thr="$5"
    PYTHONPATH=. python -m evalplus.evaluate \
        --model "$ckpt" \
        --trust_remote_code True \
        --max_new_tokens "$mnt" \
        --diffusion_steps "$dst" \
        --dataset "$dataset" \
        --backend dllm \
        --temperature 0.0 \
        --generation_method generation_multi_block \
        --alg entropy_threshold \
        --threshold "$thr" \
        --block_length 32 \
        --block_add_threshold 0.1 \
        --decoded_token_threshold 0.95 \
        --cache_delay_iter 32 \
        --early_stop True \
        --torch_compile True
}

# diffllm + humaneval/mbpp 时走 evalplus（stdout 与直接 python -m evalplus.evaluate 一致，并 tee 到 run.log）
run_diffllm_evalplus_backend() {
    local PRETRAINED_ARG="$1"
    local EP_DATASET="$2"
    local MAX_NEW_TOKENS="$3"
    local DIFFUSION_STEPS="$4"
    local THRESHOLD="$5"
    local TASK_TAG="$6"

    export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
    export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-true}"

    local EVAL_ROOT="${D3LLM_ROOT}/utils/utils_Dream/eval_instruct"
    local OUT_SLUG="${PRETRAINED_ARG//\//__}"
    OUT_SLUG="${OUT_SLUG//[^a-zA-Z0-9_.-]/_}"
    [[ ${#OUT_SLUG} -gt 96 ]] && OUT_SLUG="${OUT_SLUG:0:96}"

    local OUTPUT_DIR
    if [[ -n "${EVAL_OUTPUT_DIR:-}" ]]; then
        OUTPUT_DIR="${EVAL_OUTPUT_DIR}"
    else
        OUTPUT_DIR="${EVAL_ROOT}/eval_tmp/merged_${TASK_TAG}_m${MAX_NEW_TOKENS}_s${DIFFUSION_STEPS}_t${THRESHOLD}__${OUT_SLUG}__evalplus"
    fi
    mkdir -p "${OUTPUT_DIR}"
    local LOG_FILE="${OUTPUT_DIR}/run.log"

    local EVALPLUS_ROOT="${D3LLM_ROOT}/utils/utils_DreamCoder/code_eval/evalplus"
    if [[ ! -d "$EVALPLUS_ROOT" ]]; then
        echo "错误: 未找到 evalplus 目录: $EVALPLUS_ROOT" >&2
        exit 1
    fi

    echo "diffllm 后端: evalplus（与 dream-coder.sh 中 evalplus.evaluate multi_block 一致）"
    echo "evalplus dataset: ${EP_DATASET}（对应 lm_eval 任务名: ${TASK_TAG}）"
    echo "max_new_tokens = ${MAX_NEW_TOKENS}, diffusion_steps = ${DIFFUSION_STEPS}, threshold = ${THRESHOLD}"
    echo "pretrained: ${PRETRAINED_ARG}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
    echo "结果与完整终端输出保存: ${LOG_FILE}"

    (
        cd "$EVALPLUS_ROOT"
        _dream_coder_evalplus_multi_block "$PRETRAINED_ARG" "$EP_DATASET" "$MAX_NEW_TOKENS" "$DIFFUSION_STEPS" "$THRESHOLD"
    ) 2>&1 | tee "${LOG_FILE}"

    echo
    echo "评测完成（evalplus），正在从日志提取指标…"
    local AVG_TPF="N/A"
    local TPF_LINE
    TPF_LINE="$(grep 'Tokens per forward' "${LOG_FILE}" | tail -n 1 || true)"
    [[ -n "${TPF_LINE}" ]] && AVG_TPF="$(echo "${TPF_LINE}" | awk '{print $4}')"

    local PASS_SNIP
    PASS_SNIP="$(grep -iE 'pass@|pass @' "${LOG_FILE}" | tail -n 3 || true)"

    echo "=========== 汇总 (evalplus 后端) =========="
    echo "任务(lm_eval 名): ${TASK_TAG}"
    echo "evalplus dataset: ${EP_DATASET}"
    echo "max_new_tokens: ${MAX_NEW_TOKENS}"
    echo "diffusion_steps: ${DIFFUSION_STEPS}"
    echo "threshold: ${THRESHOLD}"
    echo "pretrained: ${PRETRAINED_ARG}"
    echo "平均 TPF (若日志中有): ${AVG_TPF}"
    if [[ -n "${PASS_SNIP}" ]]; then
        echo "pass@ 相关行（摘录）:"
        echo "${PASS_SNIP}"
    else
        echo "pass@: 请查看上方完整输出或 evalplus 结果目录"
    fi

    local SUMMARY_FILE="${OUTPUT_DIR}/eval_metrics_summary.txt"
    {
        echo "backend=evalplus"
        echo "pretrained=${PRETRAINED_ARG}"
        echo "lm_eval_task_name=${TASK_TAG}"
        echo "evalplus_dataset=${EP_DATASET}"
        echo "max_new_tokens=${MAX_NEW_TOKENS} diffusion_steps=${DIFFUSION_STEPS} threshold=${THRESHOLD}"
        echo "tokens_per_forward=${AVG_TPF}"
        echo "decode=evalplus multi_block (dream-coder.sh 同款)"
        [[ -n "${PASS_SNIP}" ]] && echo "--- pass@ excerpt ---" && echo "${PASS_SNIP}"
    } > "${SUMMARY_FILE}"
    echo "指标摘要已写入: ${SUMMARY_FILE}"
}

# ---------- 子命令: diffllm（lm_eval 或与 dream-coder 一致的 evalplus）----------
run_diffllm_mode() {
    if [[ $# -lt 2 ]]; then
        echo "用法: $0 diffllm <模型路径或HF id> <lm_eval_task> [max_new_tokens] [diffusion_steps] [threshold]" >&2
        exit 1
    fi

    local MODEL_REF="$1"
    local TASK_NAME="$2"
    shift 2
    local MAX_NEW_TOKENS="${1:-256}"
    local DIFFUSION_STEPS="${2:-256}"
    local THRESHOLD="${3:-0.4}"

    local PRETRAINED_ARG
    PRETRAINED_ARG="$(resolve_pretrained "$MODEL_REF")" || exit 1

    case "$TASK_NAME" in
        humaneval) TASK_NAME=humaneval_instruct ;;
        mbpp) TASK_NAME=mbpp_instruct ;;
    esac

    local BACKEND="${DIFFLLM_BACKEND:-auto}"
    case "$BACKEND" in
        auto)
            case "$TASK_NAME" in
                humaneval_instruct|mbpp_instruct) BACKEND=evalplus ;;
                *) BACKEND=lm_eval ;;
            esac
            ;;
        lm_eval|evalplus) ;;
        *)
            echo "错误: DIFFLLM_BACKEND 须为 auto、lm_eval 或 evalplus（当前: ${BACKEND}）" >&2
            exit 1
            ;;
    esac

    if [[ "$BACKEND" == "evalplus" ]]; then
        local EP_DATASET=""
        case "$TASK_NAME" in
            humaneval_instruct) EP_DATASET=humaneval ;;
            mbpp_instruct) EP_DATASET=mbpp ;;
            *)
                echo "错误: DIFFLLM_BACKEND=evalplus 仅适用于 humaneval_instruct / mbpp_instruct（或 humaneval / mbpp 简写）" >&2
                exit 1
                ;;
        esac
        if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
            export CUDA_VISIBLE_DEVICES=0
        fi
        run_diffllm_evalplus_backend "$PRETRAINED_ARG" "$EP_DATASET" "$MAX_NEW_TOKENS" "$DIFFUSION_STEPS" "$THRESHOLD" "$TASK_NAME"
        return
    fi

    if [[ "$TASK_NAME" == "mbpp_instruct" ]] && [[ "${NUM_FEWSHOT-unset}" == "unset" ]]; then
        export NUM_FEWSHOT=4
    fi
    export NUM_FEWSHOT="${NUM_FEWSHOT:-0}"

    # 多进程数：显式 ACCELERATE_NUM_PROCESSES 优先；否则若用户已指定多张可见 GPU，则进程数=GPU 数（与 accelerate 数据并行一致）
    local NUM_PROC
    if [[ -n "${ACCELERATE_NUM_PROCESSES:-}" ]]; then
        NUM_PROC="${ACCELERATE_NUM_PROCESSES}"
    elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && [[ "${CUDA_VISIBLE_DEVICES}" == *","* ]]; then
        NUM_PROC=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
    else
        NUM_PROC=1
    fi

    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        if [[ "$NUM_PROC" -eq 1 ]]; then
            export CUDA_VISIBLE_DEVICES=0
        else
            # 未指定可见设备但要多进程时，默认使用前 NUM_PROC 张卡
            export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_PROC - 1)))"
        fi
    fi

    export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
    export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-true}"

    local EVAL_ROOT="${D3LLM_ROOT}/utils/utils_Dream/eval_instruct"
    cd "${EVAL_ROOT}"

    local OUT_SLUG="${PRETRAINED_ARG//\//__}"
    OUT_SLUG="${OUT_SLUG//[^a-zA-Z0-9_.-]/_}"
    [[ ${#OUT_SLUG} -gt 96 ]] && OUT_SLUG="${OUT_SLUG:0:96}"

    local OUTPUT_DIR
    if [[ -n "${EVAL_OUTPUT_DIR:-}" ]]; then
        OUTPUT_DIR="${EVAL_OUTPUT_DIR}"
    else
        OUTPUT_DIR="${EVAL_ROOT}/eval_tmp/merged_${TASK_NAME}_m${MAX_NEW_TOKENS}_s${DIFFUSION_STEPS}_t${THRESHOLD}__${OUT_SLUG}"
    fi
    mkdir -p "${OUTPUT_DIR}"

    local LOG_FILE="${OUTPUT_DIR}/run.log"

    echo "运行任务: ${TASK_NAME}"
    echo "max_new_tokens = ${MAX_NEW_TOKENS}, diffusion_steps = ${DIFFUSION_STEPS}, threshold = ${THRESHOLD}, num_fewshot = ${NUM_FEWSHOT}"
    echo "pretrained: ${PRETRAINED_ARG}"
    echo "accelerate num_processes: ${NUM_PROC}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "结果目录: ${OUTPUT_DIR}"

    accelerate launch --num_processes "${NUM_PROC}" --main_process_port 46667 -m lm_eval \
        --model diffllm \
        --model_args "torch_compile=False,pretrained=${PRETRAINED_ARG},trust_remote_code=True,max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},dtype=bfloat16,temperature=0.,alg=entropy_threshold,dParallel=False,threshold=${THRESHOLD},generation_method=generation_multi_block,block_add_threshold=0.1,decoded_token_threshold=0.95,block_length=32,cache_delay_iter=1,refresh_interval=10000,early_stop=True" \
        --tasks "${TASK_NAME}" \
        --device cuda \
        --batch_size 1 \
        --num_fewshot "${NUM_FEWSHOT}" \
        --output_path "${OUTPUT_DIR}" \
        --log_samples \
        --confirm_run_unsafe_code \
        --apply_chat_template 2>&1 | tee "${LOG_FILE}"

    echo
    echo "评测完成，正在解析 TPF 和准确率..."

    local TPF_LINE AVG_TPF
    TPF_LINE="$(grep 'Tokens per forward' "${LOG_FILE}" | tail -n 1 || true)"
    if [[ -z "${TPF_LINE}" ]]; then
        echo "警告: 未在日志中找到 'Tokens per forward' 行，无法解析 TPF。" >&2
        AVG_TPF="N/A"
    else
        AVG_TPF="$(echo "${TPF_LINE}" | awk '{print $4}')"
    fi

    local PYTHON_BIN="${PYTHON_BIN:-python}"
    local ACC_OUTPUT
    ACC_OUTPUT="$(
        OUTPUT_DIR="${OUTPUT_DIR}" TASK_NAME="${TASK_NAME}" "${PYTHON_BIN}" - <<'PY'
import json, os, glob
output_dir = os.environ["OUTPUT_DIR"]
task_name = os.environ["TASK_NAME"]
pattern = os.path.join(output_dir, "**", "results_*.json")
files = glob.glob(pattern, recursive=True)
if not files:
    print("N/A")
    raise SystemExit(0)
files.sort(key=os.path.getmtime)
results_path = files[-1]
with open(results_path, "r", encoding="utf-8") as f:
    data = json.load(f)
task_res = data.get("results", {}).get(task_name)
if not task_res:
    print("N/A")
    raise SystemExit(0)
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
    print("{:.6f}".format(acc))
PY
    )"

    echo "=========== 汇总 =========="
    echo "任务: ${TASK_NAME}"
    echo "max_new_tokens: ${MAX_NEW_TOKENS}"
    echo "diffusion_steps: ${DIFFUSION_STEPS}"
    echo "threshold: ${THRESHOLD}"
    echo "num_fewshot: ${NUM_FEWSHOT}"
    echo "pretrained: ${PRETRAINED_ARG}"
    echo "平均 TPF (Tokens per forward): ${AVG_TPF}"
    echo "准确率 (exact_match, flexible-extract): ${ACC_OUTPUT}"
    echo "结果 JSON 和样本日志保存在: ${OUTPUT_DIR}"

    local SUMMARY_FILE="${OUTPUT_DIR}/eval_metrics_summary.txt"
    {
        echo "pretrained=${PRETRAINED_ARG}"
        echo "task=${TASK_NAME}"
        echo "max_new_tokens=${MAX_NEW_TOKENS} diffusion_steps=${DIFFUSION_STEPS} threshold=${THRESHOLD}"
        echo "num_fewshot=${NUM_FEWSHOT}"
        echo "tokens_per_forward=${AVG_TPF}"
        echo "exact_match_flexible_extract=${ACC_OUTPUT}"
        echo "decode=entropy_threshold+generation_multi_block block_length=32 block_add_threshold=0.1 decoded_token_threshold=0.95 cache_delay_iter=1 temperature=0"
    } > "${SUMMARY_FILE}"
    echo "指标摘要已写入: ${SUMMARY_FILE}"
}

# ---------- 子命令: evalplus ----------
run_evalplus_mode() {
    if [[ $# -lt 2 ]]; then
        echo "用法: $0 evalplus <模型路径或HF id> <humaneval|mbpp> [multi_block|vanilla]" >&2
        exit 1
    fi
    local CKPT_DIR
    CKPT_DIR="$(resolve_pretrained "$1")" || exit 1
    local DATASET="$2"
    local DECODE="${3:-multi_block}"

    if [[ "$DATASET" != "humaneval" && "$DATASET" != "mbpp" ]]; then
        echo "错误: 数据集须为 humaneval 或 mbpp" >&2
        exit 1
    fi
    if [[ "$DECODE" != "multi_block" && "$DECODE" != "vanilla" ]]; then
        echo "错误: 解码模式须为 multi_block 或 vanilla" >&2
        exit 1
    fi

    export HF_ALLOW_CODE_EVAL=1
    export HF_DATASETS_TRUST_REMOTE_CODE=true

    local EVALPLUS_ROOT="${D3LLM_ROOT}/utils/utils_DreamCoder/code_eval/evalplus"
    if [[ ! -d "$EVALPLUS_ROOT" ]]; then
        echo "错误: 未找到 evalplus 目录: $EVALPLUS_ROOT" >&2
        exit 1
    fi
    cd "$EVALPLUS_ROOT"

    local MNT="${EVALPLUS_MAX_NEW_TOKENS:-512}"
    local DST="${EVALPLUS_DIFFUSION_STEPS:-512}"
    local THR="${EVALPLUS_THRESHOLD:-0.5}"

    echo "evalplus: model=$CKPT_DIR dataset=$DATASET decode=$DECODE"
    echo "cwd=$EVALPLUS_ROOT"

    if [[ "$DECODE" == "multi_block" ]]; then
        _dream_coder_evalplus_multi_block "$CKPT_DIR" "$DATASET" "$MNT" "$DST" "$THR"
    else
        PYTHONPATH=. python -m evalplus.evaluate \
            --model "$CKPT_DIR" \
            --trust_remote_code True \
            --max_new_tokens "$MNT" \
            --diffusion_steps "$DST" \
            --dataset "$DATASET" \
            --backend dllm \
            --temperature 0.1
    fi

    echo "Dream-Coder evalplus 完成。结果见命令行输出或 evalplus 默认结果目录。"
}

# ---------- 旧版：按预设模型名 ----------
run_legacy_mode() {
    local MODEL="${1:-}"
    local DATASET="${2:-}"

    if [[ -z "$MODEL" || -z "$DATASET" ]]; then
        echo "用法: $0 <模型> <数据集>" >&2
        echo "  模型: d3llm_dream | d3llm_llada | vanilla_dream | vanilla_llada | d3llm_dream_coder | vanilla_dream_coder" >&2
        echo "  数据集: humaneval | mbpp" >&2
        echo "或: $0 diffllm <模型> <task> [m s t] ； $0 evalplus <模型> <humaneval|mbpp> ..." >&2
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
            TASK="${DATASET}_instruct"
            NUM_FEWSHOT=0
            [[ "$DATASET" == "mbpp" ]] && NUM_FEWSHOT=4
            cd "$D3LLM_ROOT/utils/utils_Dream/eval_instruct"
            if [[ "$MODEL" == "d3llm_dream" ]]; then
                if [[ -n "${D3LLM_DREAM_CHECKPOINT:-}" && "$D3LLM_DREAM_CHECKPOINT" != /* ]]; then
                    DREAM_PRETRAINED="$D3LLM_ROOT/$D3LLM_DREAM_CHECKPOINT"
                else
                    DREAM_PRETRAINED="${D3LLM_DREAM_CHECKPOINT:-d3LLM/d3LLM_Dream}"
                fi
                PYTHONPATH=. accelerate launch --main_process_port 46666 -m lm_eval --model diffllm \
                    --model_args torch_compile=False,pretrained="$DREAM_PRETRAINED",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.,alg=entropy_threshold,dParallel=False,threshold=0.5,generation_method=generation_multi_block,block_add_threshold=0.1,decoded_token_threshold=0.95,block_length=32,cache_delay_iter=10000,early_stop=True \
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
            LENGTH=256
            BLOCK_LENGTH=32
            STEPS=256
            NUM_FEWSHOT=0
            [[ "$DATASET" == "mbpp" ]] && NUM_FEWSHOT=3

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
        d3llm_dream_coder|vanilla_dream_coder)
            cd "$D3LLM_ROOT/utils/utils_DreamCoder/code_eval/evalplus"
            CKPT_DIR=""
            if [[ "$MODEL" == "d3llm_dream_coder" ]]; then
                CKPT_DIR="d3LLM/d3LLM_Dream_Coder"
            else
                CKPT_DIR="Dream-org/Dream-Coder-v0-Instruct-7B"
            fi

            echo "Running Dream-Coder evalplus for model: $CKPT_DIR on dataset: $DATASET"

            if [[ "$MODEL" == "d3llm_dream_coder" ]]; then
                PYTHONPATH=. python -m evalplus.evaluate \
                    --model "$CKPT_DIR" \
                    --trust_remote_code True \
                    --max_new_tokens 512 \
                    --diffusion_steps 512 \
                    --dataset "$DATASET" \
                    --backend dllm \
                    --temperature 0.0 \
                    --generation_method generation_multi_block \
                    --alg entropy_threshold \
                    --threshold 0.5 \
                    --block_length 32 \
                    --block_add_threshold 0.1 \
                    --decoded_token_threshold 0.95 \
                    --cache_delay_iter 32 \
                    --early_stop True \
                    --torch_compile True
            else
                PYTHONPATH=. python -m evalplus.evaluate \
                    --model "$CKPT_DIR" \
                    --trust_remote_code True \
                    --max_new_tokens 512 \
                    --diffusion_steps 512 \
                    --dataset "$DATASET" \
                    --backend dllm \
                    --temperature 0.1
            fi

            echo "Dream-Coder 评估完成（结果在 evalplus 默认输出目录，如 evalplus_results/ 或命令行输出中）。"
            ;;
        *)
            echo "错误: 未知模型 $MODEL"
            echo "  可选: d3llm_dream | d3llm_llada | vanilla_dream | vanilla_llada | d3llm_dream_coder | vanilla_dream_coder"
            echo "  或使用: $0 diffllm <模型> <task> ... | $0 evalplus <模型> <数据集> ..."
            exit 1
            ;;
    esac
}

# ---------- 主入口 ----------
case "${1:-}" in
    diffllm|difflm)
        shift
        run_diffllm_mode "$@"
        ;;
    evalplus)
        shift
        run_evalplus_mode "$@"
        ;;
    -h)
        usage
        exit 0
        ;;
    --help)
        usage
        exit 0
        ;;
    help)
        usage
        exit 0
        ;;
    "")
        usage
        exit 1
        ;;
    *)
        run_legacy_mode "$@"
        ;;
esac
