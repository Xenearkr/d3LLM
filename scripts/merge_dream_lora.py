#!/usr/bin/env python3
"""
将 d3LLM-DREAM 训练得到的 LoRA checkpoint 与 base 模型合并，保存为完整模型目录，
便于后续用 lm_eval / chat / measure_tpf_tps 等直接加载（无需 PEFT）。

用法:
  # 使用默认 base 与最新 checkpoint，输出到 output_model/d3LLM_DREAM_merged
  python scripts/merge_dream_lora.py

  # 指定 checkpoint 与输出目录
  python scripts/merge_dream_lora.py \
    --checkpoint_dir output_model/d3LLM_DREAM_local_0312_043946/checkpoint-5742 \
    --output_dir output_model/d3LLM_DREAM_merged

  # 指定 base 模型（需与训练时一致）
  python scripts/merge_dream_lora.py \
    --base_model Dream-org/Dream-v0-Instruct-7B \
    --checkpoint_dir output_model/d3LLM_DREAM_local_0312_043946/checkpoint-5742 \
    --output_dir output_model/d3LLM_DREAM_merged

依赖: 在仓库根目录执行；需安装 transformers, peft, torch。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 保证从仓库根目录运行时可导入
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from types import MethodType
from transformers import AutoModel, AutoTokenizer, GenerationConfig
from peft import PeftModel


def main(
    base_model: str = "Dream-org/Dream-v0-Instruct-7B",
    # 请将下行修改为你自己的 checkpoint 目录
    checkpoint_dir: str = "output_model/d3LLM_DREAM_local_0312_043946/checkpoint-5742",
    output_dir: str = "output_model/d3LLM_DREAM_merged",
    trust_remote_code: bool = True,
    torch_dtype: str = "bfloat16",
) -> None:
    checkpoint_path = Path(checkpoint_dir)
    out_path = Path(output_dir)

    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint 目录不存在: {checkpoint_path}")

    adapter_config = checkpoint_path / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(
            f"Checkpoint 中未找到 adapter_config.json，请确认是 PEFT LoRA 目录: {checkpoint_path}"
        )

    dtype = getattr(torch, torch_dtype, torch.bfloat16)
    print(f"加载 base 模型: {base_model}")
    model = AutoModel.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # DreamModel 没有 prepare_inputs_for_generation，PEFT 加载时会访问该方法，先挂一个占位实现
    if not hasattr(model, "prepare_inputs_for_generation"):

        def _dummy_prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {"input_ids": input_ids, **kwargs}

        model.prepare_inputs_for_generation = MethodType(
            _dummy_prepare_inputs_for_generation, model
        )

    print(f"加载 LoRA 权重: {checkpoint_path}")
    model = PeftModel.from_pretrained(model, str(checkpoint_path), is_trainable=False)
    print("合并 LoRA 到 base 并卸载 adapter...")
    model = model.merge_and_unload()

    out_path.mkdir(parents=True, exist_ok=True)
    # DreamGenerationConfig 序列化时会带上不可 JSON 的 validate，且 validate(strict=True) 不兼容。
    # 临时换成用 to_dict() 构造的标准 GenerationConfig，保存后再恢复。
    saved_generation_configs = []
    for m in (model, getattr(model, "base_model", None)):
        if m is not None and getattr(m, "generation_config", None) is not None:
            old_gc = m.generation_config
            try:
                d = old_gc.to_dict()
            except Exception:
                d = {}
            # 只保留标准 GenerationConfig 能接受的、可序列化的键
            standard_keys = set(GenerationConfig().to_dict().keys())
            d = {k: v for k, v in d.items() if k in standard_keys}
            # 通过 strict 校验：含 temperature/top_p/top_k 时需 do_sample=True（缺省为 None/False 也会触发校验报错）
            if any(d.get(k) is not None for k in ("temperature", "top_p", "top_k")) and d.get("do_sample") is not True:
                d["do_sample"] = True
            saved_generation_configs.append((m, old_gc))
            m.generation_config = GenerationConfig.from_dict(d)

    print(f"保存合并后的模型与 tokenizer 到: {out_path}")
    model.save_pretrained(out_path, safe_serialization=True)
    for m, old_gc in saved_generation_configs:
        m.generation_config = old_gc

    tokenizer.save_pretrained(out_path)
    print("完成。可直接用以下路径做评估或对话:")
    print(f"  --model_name {out_path.absolute()}")
    print(f"  或 pretrained={out_path.absolute()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 d3LLM-DREAM LoRA checkpoint 与 base 合并为完整模型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Dream-org/Dream-v0-Instruct-7B",
        help="Base 模型名或路径（需与训练时一致）",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="output_model/d3LLM_DREAM_local_0312_043946/checkpoint-5742",
        help="LoRA checkpoint 目录（含 adapter_config.json / adapter_model.safetensors）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_model/d3LLM_DREAM_merged",
        help="合并后模型保存目录",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="从 Hub 加载时信任自定义代码",
    )
    parser.add_argument(
        "--no_trust_remote_code",
        action="store_false",
        dest="trust_remote_code",
        help="禁用 trust_remote_code",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=("float32", "float16", "bfloat16"),
        help="模型权重 dtype",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        base_model=args.base_model,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.torch_dtype,
    )
