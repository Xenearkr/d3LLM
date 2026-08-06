#!/usr/bin/env python3
"""
将 base 模型与本地训练的 LoRA checkpoint 合并，并保存为完整模型目录。
合并后的模型可直接用于 eval_scripts

推荐用法（在项目根目录 d3LLM 下执行）:

  # 示例：将当前的 训练结果 checkpoint-<num> 合并
  python model_merged.py \
    --base <base_model_name> \
    --lora_path output_model/<train_name>/checkpoint-<num> \
    --output_dir output_model/<train_name> \
    --model_type <model_type>


    下面的命令是本次实验便于复制到命令行的命令
    python model_merged.py \
    --base GSAI-ML/LLaDA-8B-Instruct \
    --lora_path output_model/d3LLM_LLaDA_local_0331_031307/checkpoint-8670 \
    --output_dir output_model/d3LLM_LLaDA_merged_0331_031307 \
    --model_type normal
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge base model with LoRA checkpoint and save full model."
    )
    parser.add_argument(
        "--base",
        type=str,
        help="Base model: HF repo id or local path.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        help="Path to LoRA checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
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
    parser.add_argument(
        "--model_type",
        type=str,
        default="normal",
        choices=["causal", "normal"],
        help="determine whether use the AutoModelForCausalLM",
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

    print("Loading base model...")
    if args.model_type == "causal":
        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=dtype,
        )
    else:
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

