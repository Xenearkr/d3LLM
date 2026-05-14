"""
d3LLM 系列交互对话：生成路径与 HumanEval（EvalPlus + dllm + generation_multi_block）rollout 对齐，
突出多块扩散 + KV 延迟缓存的加速特性。

与 chat/chat_d3llm_dream_coder.py（AutoModel + 全序列 entropy + steps=256）不同：
  此处使用 utils_Dream.DreamModel + d3llm_dream_generate_util.generate_multi_block，
  默认超参与 run_code_eval.sh 中 dream_coder__humaneval_d3llm_multiblock 一致。

用法示例：
  CUDA_VISIBLE_DEVICES=0 python chat/chat_d3llm_rollout_chat.py --model d3LLM/d3LLM_Dream_Coder
  CUDA_VISIBLE_DEVICES=2 python chat/chat_d3llm_rollout_chat.py --model XenonGas/finetune_d3LLM_DREAM_Coder --warmup 1

多行提示词：在「You:」后只输入一行 <<< 回车，粘贴题目，再单独一行输入 >>> 回车结束（结束标记可改用环境变量 D3LLM_MULTILINE_END）。

环境变量（可选）：
  HF_ENDPOINT / HF_DISABLE_SSL_VERIFY 等同仓库其它脚本。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import types
from pathlib import Path
from typing import Optional

import torch
import tqdm
from transformers import AutoTokenizer

# 项目根目录
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.utils_Dream.model.modeling_dream import DreamModel
from utils.utils_Dream.model.configuration_dream import DreamConfig
from d3llm.d3llm_DREAM.d3llm_dream_generate_util import DreamGenerationMixin as D3LLMGenerationMixin

# 与 eval_scripts/run_code_eval.sh dream_coder__humaneval_d3llm_multiblock 对齐
PRESET_HUMANEVAL_CODER = {
    "attention_mask": None,
    "max_new_tokens": 512,
    "output_history": False,
    "return_dict_in_generate": True,
    "steps": 512,
    "temperature": 0.0,
    "alg": "entropy_threshold",
    "threshold": 0.5,
    "block_size": 32,
    "block_add_threshold": 0.1,
    "decoded_token_threshold": 0.97,
    "cache_delay_iter": 32,
    "early_stop": True,
}

# chat_d3llm_dream.py 原版（偏通用对话、无 KV 分支）
PRESET_CHAT_DREAM = {
    **{k: v for k, v in PRESET_HUMANEVAL_CODER.items() if k != "decoded_token_threshold"},
    "decoded_token_threshold": 0.95,
    "cache_delay_iter": 10000,
    "max_new_tokens": 256,
    "steps": 256,
}


def _maybe_disable_ssl_verify():
    if os.environ.get("HF_DISABLE_SSL_VERIFY", "").lower() not in ("1", "true", "yes"):
        return
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
    import requests

    _orig_send = requests.Session.send

    def _send_no_verify(self, request, **kwargs):
        kwargs["verify"] = False
        return _orig_send(self, request, **kwargs)

    requests.Session.send = _send_no_verify


def build_gen_kwargs(args: argparse.Namespace) -> dict:
    if args.preset == "humaneval_coder":
        base = dict(PRESET_HUMANEVAL_CODER)
    else:
        base = dict(PRESET_CHAT_DREAM)
    base["max_new_tokens"] = args.max_new_tokens
    base["steps"] = args.steps
    base["threshold"] = args.threshold
    base["block_size"] = args.block_size
    base["cache_delay_iter"] = args.cache_delay_iter
    base["decoded_token_threshold"] = args.decoded_token_threshold
    return base


def extract_assistant_text(
    tokenizer,
    input_ids: torch.Tensor,
    sequences: torch.Tensor,
) -> str:
    """
    先对「仅生成段」解码（与 rollout 一致）；若为空或极短，再用全序列减去 prompt 还原，
    避免 chat_d3llm_dream_coder 里仅 split(eos) 导致空白的问题。
    """
    prompt_len = input_ids.shape[1]
    gen_ids = sequences[0, prompt_len:]
    new_only = tokenizer.decode(gen_ids, skip_special_tokens=True)
    new_only = new_only.strip()
    if new_only and len(new_only) > 0:
        eos = tokenizer.eos_token
        if eos and eos in new_only:
            new_only = new_only.split(eos)[0].strip()
        return new_only

    full = tokenizer.decode(sequences[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    if full.startswith(prompt_text):
        tail = full[len(prompt_text) :].strip()
    else:
        tail = new_only
    eos = tokenizer.eos_token
    if eos and tail and eos in tail:
        tail = tail.split(eos)[0].strip()
    for marker in ("\nassistant\n", "assistant\n", "Assistant:", "<|im_start|>assistant"):
        if marker in full and marker != prompt_text:
            part = full.split(marker)[-1]
            if eos and eos in part:
                part = part.split(eos)[0]
            part = part.strip()
            if part:
                tail = part
                break
    return tail


def read_user_message(multiline_start: str, multiline_end: str) -> Optional[str]:
    """
    单行：在 You: 后直接输入一整句。
    多行：第一行必须为单独的起始标记（默认 <<<），随后任意行，直到单独一行的结束标记（默认 >>>）。
    返回 None 表示退出对话；返回空串表示本轮跳过。
    """
    first = input("\nYou: ")
    if first.strip().lower() in ("quit", "exit", "q"):
        return None
    if first.strip() != multiline_start:
        return first.strip()

    print(
        f"多行模式：粘贴内容；单独一行输入 {multiline_end!r} 结束。"
        f"（若首行误进多行模式，可先 {multiline_end} 空结束再重来）",
        file=sys.stderr,
    )
    lines: list[str] = []
    while True:
        line = input()
        if line.rstrip("\n").strip() == multiline_end:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def parse_args():
    p = argparse.ArgumentParser(description="d3LLM rollout-style chat (multiblock KV)")
    p.add_argument(
        "--model",
        type=str,
        default=os.environ.get("D3LLM_MODEL", "d3LLM/d3LLM_Dream_Coder"),
        help="HF 模型 id 或本地目录（Dream / Dream-Coder 系列，需含 DreamConfig）",
    )
    p.add_argument(
        "--preset",
        choices=("humaneval_coder", "chat_dream"),
        default="humaneval_coder",
        help="humaneval_coder：与 dream_coder HumanEval multiblock 评测一致；chat_dream：对齐 chat_d3llm_dream.py",
    )
    p.add_argument("--warmup", type=int, default=2, help="预热轮数（默认 2，设为 0 跳过）")
    p.add_argument("--max-new-tokens", type=int, dest="max_new_tokens", default=None)
    p.add_argument("--steps", type=int, default=None, help="扩散步数，默认随 preset")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--block-size", type=int, dest="block_size", default=None)
    p.add_argument("--cache-delay-iter", type=int, dest="cache_delay_iter", default=None)
    p.add_argument(
        "--decoded-token-threshold",
        type=float,
        dest="decoded_token_threshold",
        default=None,
    )
    p.add_argument(
        "--torch-compile",
        action="store_true",
        help="尝试 torch.compile（首次很慢；与 EvalPlus 脚本一致可选）",
    )
    p.add_argument(
        "--flash-attn",
        action="store_true",
        help="在 config 上启用 flash_attention_2（需硬件/包支持）",
    )
    p.add_argument(
        "--ml-start",
        default=os.environ.get("D3LLM_MULTILINE_START", "<<<"),
        metavar="STR",
        help="多行模式起始行（整行仅含此串，默认 <<<，可用环境变量 D3LLM_MULTILINE_START）",
    )
    p.add_argument(
        "--ml-end",
        default=os.environ.get("D3LLM_MULTILINE_END", ">>>"),
        metavar="STR",
        help="多行模式结束行（单独一行，默认 >>>，可用 D3LLM_MULTILINE_END）",
    )
    return p.parse_args()


def main():
    _maybe_disable_ssl_verify()
    args = parse_args()

    # 默认数值回填到 argparse.Namespace，供 build_gen_kwargs 使用
    if args.preset == "humaneval_coder":
        d = PRESET_HUMANEVAL_CODER
    else:
        d = PRESET_CHAT_DREAM
    if args.max_new_tokens is None:
        args.max_new_tokens = d["max_new_tokens"]
    if args.steps is None:
        args.steps = d["steps"]
    if args.threshold is None:
        args.threshold = d["threshold"]
    if args.block_size is None:
        args.block_size = d["block_size"]
    if args.cache_delay_iter is None:
        args.cache_delay_iter = d["cache_delay_iter"]
    if args.decoded_token_threshold is None:
        args.decoded_token_threshold = d["decoded_token_threshold"]

    if args.max_new_tokens % args.block_size != 0:
        raise SystemExit(
            f"max_new_tokens ({args.max_new_tokens}) 必须能被 block_size ({args.block_size}) 整除"
            "（与多 block 扩散实现一致）。请调整 --max-new-tokens 或 --block-size。"
        )

    gen_kwargs = build_gen_kwargs(args)

    model_path = args.model.strip()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Loading DreamModel from {model_path} (preset={args.preset}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model_config = DreamConfig.from_pretrained(model_path, trust_remote_code=True)
    if args.flash_attn:
        try:
            model_config._attn_implementation = "flash_attention_2"
            print("Using flash_attention_2 in config.")
        except Exception as e:
            print(f"Warning: could not set flash_attention_2: {e}")

    model = DreamModel.from_pretrained(
        model_path,
        config=model_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = model.to(device).eval()

    model.generate_multi_block = types.MethodType(D3LLMGenerationMixin.generate_multi_block, model)
    model._sample_multi_block = types.MethodType(D3LLMGenerationMixin._sample_multi_block, model)
    model._sample_multi_block_kv_cache = types.MethodType(
        D3LLMGenerationMixin._sample_multi_block_kv_cache, model
    )
    model._prepare_inputs = types.MethodType(D3LLMGenerationMixin._prepare_inputs, model)

    if args.torch_compile:
        print("torch.compile(model) — 首次运行会较慢 …")
        model = torch.compile(model, mode="reduce-overhead")

    print("\n" + "=" * 80)
    print(
        f"d3LLM rollout chat | multiblock + cache_delay_iter={gen_kwargs['cache_delay_iter']} | "
        f"model={model_path}"
    )
    print("输入 quit / exit 结束。统计：TPS=生成 token/s，TPF=token/NFE（与 EvalPlus 汇报语义一致）")
    ms, me = args.ml_start.strip(), args.ml_end.strip()
    print(f"多行提示：第一行仅输入 {ms!r} 回车 → 粘贴内容 → 单独一行 {me!r} 结束。")
    print("=" * 80 + "\n")

    # 短预热：轮数少、用同套 gen 参数以尽量对齐评测图编译/缓存行为
    test_questions_path = _REPO_ROOT / "utils" / "serve" / "test_question.txt"
    try:
        content = test_questions_path.read_text()
        test_questions = [q.strip() for q in content.split("\n\n") if q.strip()]
    except Exception:
        test_questions = ["def add(a, b):\n    \"\"\"Return a + b.\"\"\""]

    n_warm = max(0, min(args.warmup, len(test_questions) * 2 or 1))
    if n_warm == 0:
        print("跳过预热。")
    else:
        for i in tqdm.tqdm(range(n_warm), desc="Warmup"):
            prompt_text = test_questions[i % len(test_questions)]
            inputs = tokenizer(prompt_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)
            gen_kw = {**gen_kwargs, "attention_mask": attention_mask}
            with torch.no_grad():
                _, _ = model.generate_multi_block(input_ids, **gen_kw)
        print("Warmup done.\n")

    messages: list[dict] = []
    while True:
        user_input = read_user_message(ms, me)
        if user_input is None:
            print("Goodbye!")
            break
        if not user_input:
            continue

        messages = [{"role": "user", "content": user_input}]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)

        gen_kw = {**gen_kwargs, "attention_mask": attention_mask}
        t0 = time.perf_counter()
        with torch.no_grad():
            output, nfe = model.generate_multi_block(input_ids, **gen_kw)
        elapsed = time.perf_counter() - t0

        assistant_text = extract_assistant_text(tokenizer, input_ids, output.sequences)
        num_tokens = len(tokenizer.encode(assistant_text, add_special_tokens=False))
        tps = num_tokens / elapsed if elapsed > 0 else 0.0
        tpf = num_tokens / nfe if nfe and nfe > 0 else 0.0

        print("\n\033[34mAssistant:\033[0m")
        print("\033[34m" + (assistant_text or "[empty — try lowering decoded_token_threshold or check prompt]") + "\033[0m")
        print(
            f"\n[Stats] tokens={num_tokens} | time={elapsed:.2f}s | NFE={nfe} | "
            f"TPS={tps:.2f} tok/s | TPF={tpf:.4f} tok/forward"
        )


if __name__ == "__main__":
    main()
