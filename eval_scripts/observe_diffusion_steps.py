#!/usr/bin/env python3
"""
在 HumanEval / MBPP 上跑 Dream 或 d3LLM-Dream，观察不同 diffusion steps 的生成过程。

用法:
  # Dream，HumanEval，对比 steps=8/32/64/128/256，并保存某次生成的逐步历史
  python observe_diffusion_steps.py --model_type dream --dataset humaneval --save_history

  # 只对比不同 steps，不保存逐步历史
  python observe_diffusion_steps.py --model_type dream --dataset mbpp --steps 8 32 64 256

  # 使用 d3LLM-Dream（多块解码），对比不同 steps
  python observe_diffusion_steps.py --model_type d3llm --dataset humaneval --steps 32 128 256

依赖: 在 d3LLM 仓库根目录执行，且已安装 datasets, torch, transformers。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 保证从仓库根目录运行，便于导入 utils
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def get_prompts(dataset_name: str, num_samples: int = 3):
    """加载 HumanEval 或 MBPP 的 prompt（用于代码补全的输入）。"""
    if dataset_name.lower() == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
        # HumanEval: 每条约有 "prompt"（函数签名+docstring，要模型补全）
        prompts = [ds[i]["prompt"] for i in range(min(num_samples, len(ds)))]
        task_ids = [ds[i].get("task_id", str(i)) for i in range(min(num_samples, len(ds)))]
    elif dataset_name.lower() == "mbpp":
        ds = load_dataset("mbpp", "full", split="test")
        # MBPP: "text" 为题目描述，可简单包装成指令
        prompts = []
        task_ids = []
        for i in range(min(num_samples, len(ds))):
            text = ds[i]["text"]
            prompt = f"Write a Python function for the following task.\n{text}"
            prompts.append(prompt)
            task_ids.append(ds[i].get("task_id", str(i)))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use humaneval or mbpp.")
    return prompts, task_ids


def load_dream_vanilla(model_name: str = "Dream-org/Dream-v0-Instruct-7B", device: str = "cuda:0"):
    """加载 Vanilla Dream，用于带 diffusion_steps 的 diffusion_generate。"""
    from utils.utils_Dream.model.modeling_dream import DreamModel
    from utils.utils_Dream.model.configuration_dream import DreamConfig
    from utils.utils_Dream.model.generation_utils import DreamGenerationMixin, DreamGenerationConfig
    import types

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = DreamConfig.from_pretrained(model_name, trust_remote_code=True)
    model = DreamModel.from_pretrained(
        model_name, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model = model.to(device).eval()
    model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model)
    model._sample = types.MethodType(DreamGenerationMixin._sample, model)
    return model, tokenizer, DreamGenerationConfig


def load_d3llm_dream(model_name: str = "d3LLM/d3LLM_Dream", device: str = "cuda:0"):
    """加载 d3LLM-Dream（多块解码）。"""
    from utils.utils_Dream.model.modeling_dream import DreamModel
    from utils.utils_Dream.model.configuration_dream import DreamConfig
    from d3llm.d3llm_DREAM.d3llm_dream_generate_util import DreamGenerationMixin as D3LLMGenerationMixin
    import types

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = DreamConfig.from_pretrained(model_name, trust_remote_code=True)
    model = DreamModel.from_pretrained(
        model_name, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model = model.to(device).eval()
    model.generate_multi_block = types.MethodType(D3LLMGenerationMixin.generate_multi_block, model)
    model._sample_multi_block = types.MethodType(D3LLMGenerationMixin._sample_multi_block, model)
    model._sample_multi_block_kv_cache = types.MethodType(D3LLMGenerationMixin._sample_multi_block_kv_cache, model)
    model._prepare_inputs = types.MethodType(D3LLMGenerationMixin._prepare_inputs, model)
    return model, tokenizer, None


def run_dream_vanilla(
    model, tokenizer, DreamGenerationConfig,
    prompt: str, steps: int, max_new_tokens: int = 256,
    output_history: bool = False, device: str = "cuda:0",
):
    """Vanilla Dream：一次生成，指定 steps，可选返回每步历史。"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    input_ids = text.input_ids.to(device)
    attention_mask = text.attention_mask.to(device) if hasattr(text, "attention_mask") else None

    gen_config = DreamGenerationConfig(
        max_length=input_ids.shape[1] + max_new_tokens,
        mask_token_id=model.config.mask_token_id,
        steps=steps,
        temperature=0.1,
        top_p=0.9,
        alg="entropy",
        output_history=output_history,
        return_dict_in_generate=True,
    )

    with torch.no_grad():
        out, nfe = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            generation_config=gen_config,
        )

    if hasattr(out, "sequences"):
        seq = out.sequences
        history = getattr(out, "history", None)
    else:
        seq = out
        history = None

    response_ids = seq[0][input_ids.shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response, nfe, (history if output_history and history else None)


def run_d3llm_dream(
    model, tokenizer, _,
    prompt: str, steps: int, max_new_tokens: int = 256,
    output_history: bool = False, device: str = "cuda:0",
):
    """d3LLM-Dream：多块解码。注意：多块路径内部以「跑满直到无 MASK」结束，未使用 steps 上限，故 NFE/耗时与 steps 参数无关。"""
    from d3llm.d3llm_DREAM.d3llm_dream_generate_util import DreamGenerationConfig

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    input_ids = text.input_ids.to(device)

    gen_config = DreamGenerationConfig(
        max_length=input_ids.shape[1] + max_new_tokens,
        mask_token_id=model.config.mask_token_id,
        steps=steps,
        temperature=0.0,
        alg="entropy_threshold",
        return_dict_in_generate=True,
    )

    with torch.no_grad():
        result, nfe = model.generate_multi_block(
            input_ids,
            generation_config=gen_config,
            threshold=0.5,
            block_size=32,
            block_add_threshold=0.1,
            decoded_token_threshold=0.95,
            cache_delay_iter=10000,
            early_stop=True,
        )

    if hasattr(result, "sequences"):
        seq = result.sequences
        history = getattr(result, "history", None)
    else:
        seq = result
        history = None

    response_ids = seq[0][input_ids.shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response, nfe, (history if output_history and history else None)


def main():
    parser = argparse.ArgumentParser(description="Observe diffusion steps on HumanEval/MBPP")
    parser.add_argument("--model_type", type=str, default="dream", choices=["dream", "d3llm"],
                        help="dream: Vanilla Dream (steps 生效); d3llm: d3LLM-Dream (多块解码，steps 不生效，NFE 由收敛决定)")
    parser.add_argument("--dataset", type=str, default="humaneval", choices=["humaneval", "mbpp"])
    parser.add_argument("--num_samples", type=int, default=2, help="Number of prompts to try")
    parser.add_argument("--steps", type=int, nargs="+", default=[8, 32, 64, 128, 256],
                        help="Diffusion steps to compare")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--save_history", action="store_true",
                        help="Run one prompt with output_history and save per-step decoding")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Default: eval_scripts/observe_steps_output/")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    output_dir = args.output_dir or str(REPO_ROOT / "eval_scripts" / "observe_steps_output")
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    prompts, task_ids = get_prompts(args.dataset, args.num_samples)
    print(f"Loaded {len(prompts)} prompts from {args.dataset}")

    # 加载模型
    model_name = "Dream-org/Dream-v0-Instruct-7B" if args.model_type == "dream" else "d3LLM/d3LLM_Dream"
    print(f"Loading {args.model_type} model: {model_name} ...")
    if args.model_type == "dream":
        model, tokenizer, GenConfig = load_dream_vanilla(model_name, args.device)
        run_fn = run_dream_vanilla
    else:
        model, tokenizer, GenConfig = load_d3llm_dream(model_name, args.device)
        run_fn = run_d3llm_dream

    # 对比不同 steps
    all_results = []
    for sample_idx, (prompt, task_id) in enumerate(zip(prompts, task_ids)):
        row = {"task_id": task_id, "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt}
        by_steps = {}
        for steps in args.steps:
            t0 = time.perf_counter()
            response, nfe, _ = run_fn(
                model, tokenizer, GenConfig,
                prompt, steps=steps, max_new_tokens=args.max_new_tokens,
                output_history=False, device=args.device,
            )
            elapsed = time.perf_counter() - t0
            by_steps[steps] = {
                "response_preview": (response[:500] + "..." if len(response) > 500 else response),
                "response_length": len(response),
                "nfe": nfe,
                "time_sec": round(elapsed, 3),
            }
            print(f"  [{task_id}] steps={steps} -> nfe={nfe}, time={elapsed:.2f}s, len={len(response)}")
        row["by_steps"] = by_steps
        all_results.append(row)

    out_path = os.path.join(output_dir, f"compare_steps_{args.model_type}_{args.dataset}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Saved comparison to {out_path}")

    # 可选：保存某次生成的逐步历史（仅 Vanilla Dream 支持 output_history）
    if args.save_history and args.model_type == "dream" and GenConfig is not None:
        prompt, task_id = prompts[0], task_ids[0]
        steps_for_history = max(args.steps)
        response, nfe, history = run_dream_vanilla(
            model, tokenizer, GenConfig,
            prompt, steps=steps_for_history, max_new_tokens=args.max_new_tokens,
            output_history=True, device=args.device,
        )
        if history is not None:
            history_path = os.path.join(output_dir, f"step_history_{args.dataset}_{task_id.replace('/', '_')}.txt")
            messages = [{"role": "user", "content": prompt}]
            prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            prompt_len = len(tokenizer(prompt_str, add_special_tokens=False)["input_ids"])
            # 更“像 diffusion”地可视化：保留 [MASK] 位置，而不是只看确定前缀
            mask_token_id = getattr(model.config, "mask_token_id", None)
            with open(history_path, "w", encoding="utf-8") as f:
                f.write(
                    f"# task_id: {task_id}\n"
                    f"# steps: {steps_for_history}\n"
                    f"# prompt (first 300 chars):\n{prompt[:300]}...\n\n"
                    f"# NOTE: [MASK] 表示该位置在该步仍未确定，将在后续步中被填充/改写。\n\n"
                )
                for step, x in enumerate(history):
                    # x: [1, seq_len]，这里我们保留 prompt 后的所有位置，并显式标出 MASK
                    gen_ids = x[0][prompt_len:]
                    token_ids = gen_ids.tolist()
                    pieces = []
                    for tid in token_ids:
                        if mask_token_id is not None and tid == mask_token_id:
                            pieces.append("[MASK]")
                        else:
                            # 单 token 解码以保留局部结构，避免掩码被吞掉
                            pieces.append(
                                tokenizer.decode([tid], skip_special_tokens=False)
                            )
                    text = "".join(pieces)
                    f.write(f"--- Step {step + 1} ---\n{text}\n\n")
            print(f"Saved step-by-step history to {history_path}")
        else:
            print("(output_history was requested but model did not return history)")

    print("Done.")


if __name__ == "__main__":
    main()
