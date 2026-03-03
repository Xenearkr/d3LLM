#!/usr/bin/env python3
"""
测量 TPF（Tokens Per Forward）和 TPS（Tokens Per Second），支持运行时可选模型与数据集。

  TPF = 生成 token 数 / 前向次数(NFE)
  TPS = 生成 token 数 / 总耗时(秒)

用法:
  # Dream + HumanEval，默认 3 条样本
  python eval_scripts/measure_tpf_tps.py --model_type dream --dataset humaneval

  # d3LLM + MBPP，5 条样本
  python eval_scripts/measure_tpf_tps.py --model_type d3llm --dataset mbpp --num_samples 5

  # 指定 steps（dream 时生效）、最大生成长度、结果输出路径
  python eval_scripts/measure_tpf_tps.py --model_type dream --dataset humaneval --steps 64 --max_new_tokens 256 --output results.json

  # 使用本地或自定义模型路径
  python eval_scripts/measure_tpf_tps.py --model_type d3llm --model_name /path/to/d3LLM_Dream --dataset humaneval

依赖: 在 d3LLM 仓库根目录执行；复用 eval_scripts/observe_diffusion_steps.py 和 utils_LLaDA 中的逻辑。
"""

import argparse
import json
import os
import sys
import time
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPTS = Path(__file__).resolve().parent
LLADA_ROOT = REPO_ROOT / "utils" / "utils_LLaDA"

for p in (REPO_ROOT, EVAL_SCRIPTS, LLADA_ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

# 复用 observe_diffusion_steps 的数据加载与模型运行逻辑（Dream / d3LLM-Dream）
from observe_diffusion_steps import (
    get_prompts,
    load_dream_vanilla,
    load_d3llm_dream,
    run_dream_vanilla,
    run_d3llm_dream,
)

# LLaDA / d3LLM-LLaDA
from transformers import AutoTokenizer
from utils.utils_LLaDA.model.modeling_llada import LLaDAModelLM
from utils.utils_LLaDA.generate import (
    generate as llada_generate,
    generate_with_prefix_cache as llada_generate_with_prefix_cache,
)


def load_llada_model(model_name: str, device: str = "cuda:0"):
    """加载 LLaDA / d3LLM-LLaDA 模型与 tokenizer。"""
    model = LLaDAModelLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer


def run_llada(
    model,
    tokenizer,
    _,
    prompt: str,
    steps: int,
    max_new_tokens: int = 256,
    output_history: bool = False,  # 与 Dream 接口对齐，占位
    device: str = "cuda:0",
):
    """Vanilla LLaDA：使用 utils_LLaDA.generate.generate。"""
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    input_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ].to(device)

    # 与脚本中设置基本一致：block_length=32，threshold=0.5
    out_ids, nfe = llada_generate(
        model,
        input_ids,
        steps=steps,
        gen_length=max_new_tokens,
        block_length=32,
        temperature=0.0,
        remasking="low_confidence",
        threshold=0.5,
    )
    gen_ids = out_ids[:, input_ids.shape[1] :]
    response = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return response, nfe, None


def run_d3llm_llada(
    model,
    tokenizer,
    _,
    prompt: str,
    steps: int,
    max_new_tokens: int = 256,
    output_history: bool = False,
    device: str = "cuda:0",
):
    """d3LLM-LLaDA：使用带 prefix-cache 的生成近似 d3LLM-LLaDA 行为。"""
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    input_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ].to(device)

    out_ids, nfe = llada_generate_with_prefix_cache(
        model,
        input_ids,
        steps=steps,
        gen_length=max_new_tokens,
        block_length=32,
        temperature=0.0,
        remasking="low_confidence",
        threshold=0.5,
    )
    gen_ids = out_ids[:, input_ids.shape[1] :]
    response = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return response, nfe, None


def main():
    parser = argparse.ArgumentParser(
        description="Measure TPF (tokens per forward) and TPS (tokens per second) with optional model and dataset."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="dream",
        choices=["dream", "d3llm-dream", "d3llm", "llada", "d3llm-llada"],
        help=(
            "模型类型: "
            "dream=Vanilla Dream; "
            "d3llm-dream/d3llm=d3LLM-Dream; "
            "llada=LLaDA; "
            "d3llm-llada=d3LLM-LLaDA"
        ),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="可选：覆盖默认模型路径，如本地目录或 HuggingFace 模型 id",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="humaneval",
        choices=["humaneval", "mbpp"],
        help="数据集: humaneval 或 mbpp",
    )
    parser.add_argument("--num_samples", type=int, default=3, help="使用的 prompt 数量")
    parser.add_argument(
        "--steps",
        type=int,
        default=64,
        help="Diffusion steps（仅 dream 时生效；d3llm 多块解码由收敛决定）",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="可选：将汇总与逐条结果写入 JSON 文件",
    )
    args = parser.parse_args()

    model_name = args.model_name
    if model_name is None:
        if args.model_type == "dream":
            model_name = "Dream-org/Dream-v0-Instruct-7B"
        elif args.model_type in ("d3llm-dream", "d3llm"):
            model_name = "d3LLM/d3LLM_Dream"
        elif args.model_type == "llada":
            model_name = "GSAI-ML/LLaDA-8B-Instruct"
        elif args.model_type == "d3llm-llada":
            model_name = "d3LLM/d3LLM_LLaDA"
        else:
            raise ValueError(f"Unknown model_type: {args.model_type}")

    prompts, task_ids = get_prompts(args.dataset, args.num_samples)
    print(f"Loaded {len(prompts)} prompts from {args.dataset}")

    print(f"Loading {args.model_type} model: {model_name} ...")
    if args.model_type == "dream":
        model, tokenizer, GenConfig = load_dream_vanilla(model_name, args.device)
        run_fn = run_dream_vanilla
    elif args.model_type in ("d3llm-dream", "d3llm"):
        model, tokenizer, GenConfig = load_d3llm_dream(model_name, args.device)
        run_fn = run_d3llm_dream
    elif args.model_type == "llada":
        model, tokenizer = load_llada_model(model_name, args.device)
        GenConfig = None
        run_fn = run_llada
    elif args.model_type == "d3llm-llada":
        model, tokenizer = load_llada_model(model_name, args.device)
        GenConfig = None
        # 使用 vanilla generate：generate_with_prefix_cache 的 past_key_values 格式与当前 modeling 不兼容
        run_fn = run_llada
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    results = []
    for prompt, task_id in zip(prompts, task_ids):
        t0 = time.perf_counter()
        response, nfe, _ = run_fn(
            model,
            tokenizer,
            GenConfig,
            prompt,
            steps=args.steps,
            max_new_tokens=args.max_new_tokens,
            output_history=False,
            device=args.device,
        )
        elapsed = time.perf_counter() - t0
        num_tokens = len(tokenizer.encode(response, add_special_tokens=False))
        tpf = num_tokens / nfe if nfe else 0.0
        tps = num_tokens / elapsed if elapsed > 0 else 0.0
        row = {
            "task_id": task_id,
            "num_tokens": num_tokens,
            "nfe": nfe,
            "time_sec": round(elapsed, 4),
            "tpf": round(tpf, 4),
            "tps": round(tps, 4),
        }
        results.append(row)
        print(
            f"  [{task_id}] tokens={num_tokens} nfe={nfe} time={elapsed:.2f}s -> TPF={tpf:.2f} TPS={tps:.2f}"
        )

    num_tokens_list = [r["num_tokens"] for r in results]
    nfe_list = [r["nfe"] for r in results]
    time_list = [r["time_sec"] for r in results]
    tpf_list = [r["tpf"] for r in results]
    tps_list = [r["tps"] for r in results]
    n = len(results)
    mean_tpf = sum(tpf_list) / n if n else 0
    mean_tps = sum(tps_list) / n if n else 0
    var_tpf = sum((x - mean_tpf) ** 2 for x in tpf_list) / n if n else 0
    var_tps = sum((x - mean_tps) ** 2 for x in tps_list) / n if n else 0
    std_tpf = var_tpf ** 0.5
    std_tps = var_tps ** 0.5

    print("--- Summary ---")
    print(f"  Mean TPF: {mean_tpf:.4f} (±{std_tpf:.4f})")
    print(f"  Mean TPS: {mean_tps:.4f} (±{std_tps:.4f})")
    print(f"  Total tokens: {sum(num_tokens_list)}, total NFE: {sum(nfe_list)}, total time: {sum(time_list):.2f}s")

    out = {
        "model_type": args.model_type,
        "model_name": model_name,
        "dataset": args.dataset,
        "num_samples": n,
        "steps": args.steps,
        "max_new_tokens": args.max_new_tokens,
        "summary": {
            "mean_tpf": round(mean_tpf, 4),
            "std_tpf": round(std_tpf, 4),
            "mean_tps": round(mean_tps, 4),
            "std_tps": round(std_tps, 4),
            "total_tokens": sum(num_tokens_list),
            "total_nfe": sum(nfe_list),
            "total_time_sec": round(sum(time_list), 4),
        },
        "per_sample": results,
    }
    if args.output:
        out_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Saved to {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
