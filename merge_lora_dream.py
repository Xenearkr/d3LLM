#!/usr/bin/env python3
"""
将 Dream base 模型与本地训练的 LoRA checkpoint 合并，并保存为完整模型目录。
合并后的模型可直接用于 eval_scripts（思路 A：pretrained=合并后目录）。

用法（在项目根目录 d3LLM 下执行, YOUR_CHECKPOINT_PATH 需要替换为实际的 LoRA checkpoint 路径）:
  python merge_lora_dream.py \
    --base Dream-org/Dream-v0-Instruct-7B \
    --lora_path output_model/d3LLM_DREAM_local_YOUR_CHECKPOINT_PATH/checkpoint-5742\
    --output_dir output_model/merged_d3LLM_DREAM_5742

  # 使用国内镜像下载 base 时:
  HF_ENDPOINT=https://hf-mirror.com python merge_lora_dream.py --base ... --lora_path ... --output_dir ...
"""
import argparse
import sys
import types
from pathlib import Path

import torch
from transformers import AutoTokenizer
from peft import PeftModel

# 保证在项目根下可 import utils / d3llm
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.utils_Dream.model.modeling_dream import DreamModel
from utils.utils_Dream.model.configuration_dream import DreamConfig


def main():
    parser = argparse.ArgumentParser(description="Merge Dream base model with LoRA and save full model.")
    parser.add_argument("--base", type=str, default="Dream-org/Dream-v0-Instruct-7B",
                        help="Base model: HF repo id or local path.")
    parser.add_argument("--lora_path", type=str, required=True,
                        help="Path to LoRA checkpoint (e.g. output_model/.../checkpoint-5742).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save merged model and tokenizer.")
    parser.add_argument("--trust_remote_code", action="store_true", default=True,
                        help="Trust remote code for tokenizer/config.")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="dtype for merged model.")
    args = parser.parse_args()

    dtype = getattr(torch, args.torch_dtype)
    lora_path = Path(args.lora_path)
    output_dir = Path(args.output_dir)

    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA path not found: {lora_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer from base...")
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=args.trust_remote_code)

    print("Loading Dream base model...")
    model_config = DreamConfig.from_pretrained(
        args.base, trust_remote_code=args.trust_remote_code
    )
    model = DreamModel.from_pretrained(
        args.base,
        config=model_config,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
    )

    if not hasattr(model, "prepare_inputs_for_generation"):
        def _prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {"input_ids": input_ids, **kwargs}
        model.prepare_inputs_for_generation = types.MethodType(_prepare_inputs_for_generation, model)

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(lora_path))
    print("Merging LoRA into base...")
    model = model.merge_and_unload()

    print(f"Saving merged model and tokenizer to {output_dir}...")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print("Done. Use in eval with: pretrained=%s" % output_dir.resolve())


if __name__ == "__main__":
    main()