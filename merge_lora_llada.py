#!/usr/bin/env python3
"""
将 LLaDA base 模型与本地训练的 LoRA checkpoint 合并，并保存为完整模型目录。
合并后的模型可直接用于 eval_scripts（例如 run_code_eval.sh 中的 d3llm_llada 分支）。

推荐用法（在项目根目录 d3LLM 下执行）:

  # 示例：将当前的 d3LLM-LLaDA 训练结果 checkpoint-8664 合并
  python merge_lora_llada.py \
    --base GSAI-ML/LLaDA-8B-Instruct \
    --lora_path output_model/d3LLM_LLaDA_local_0310_043730/checkpoint-8664 \
    --output_dir output_model/merged_d3LLM_LLaDA_8664

  # 使用国内镜像下载 base 时:
  HF_ENDPOINT=https://hf-mirror.com python merge_lora_llada.py --base ... --lora_path ... --output_dir ...
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge LLaDA base model with LoRA checkpoint and save full model."
    )
    parser.add_argument(
        "--base",
        type=str,
        default="GSAI-ML/LLaDA-8B-Instruct",
        help="Base model: HF repo id or local path.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="output_model/d3LLM_LLaDA_local_0310_043730/checkpoint-8664",
        help="Path to LoRA checkpoint (e.g. output_model/.../checkpoint-8664).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_model/merged_d3LLM_LLaDA_8664",
        help="Directory to save merged model and tokenizer.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Trust remote code when loading base model/tokenizer.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="dtype for merged model.",
    )
    args = parser.parse_args()

    dtype = getattr(torch, args.torch_dtype)
    lora_path = Path(args.lora_path)
    output_dir = Path(args.output_dir)

    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA path not found: {lora_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer from base...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base,
        trust_remote_code=args.trust_remote_code,
    )

    print("Loading LLaDA base model...")
    model = AutoModel.from_pretrained(
        args.base,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(lora_path))

    print("Merging LoRA into base...")
    model = model.merge_and_unload()

    print(f"Saving merged model and tokenizer to {output_dir}...")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print("Done. Use in eval with, for example:")
    print(f"  pretrained={output_dir.resolve()}")


if __name__ == "__main__":
    main()

