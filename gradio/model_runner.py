"""
d3LLM Dream-Coder 模型加载与对话生成。
"""
from __future__ import annotations

import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

# 项目根目录加入 path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from d3llm.d3llm_DREAM.d3llm_dream_generate_util import (  # noqa: E402
    DreamGenerationConfig,
    DreamGenerationMixin,
)
from generation_trace import (  # noqa: E402
    sample_multi_block_with_trace,
    sample_multi_block_with_trace_stream,
)
from utils.utils_Dream.model.configuration_dream import DreamConfig  # noqa: E402
from utils.utils_Dream.model.modeling_dream import DreamModel  # noqa: E402

DEFAULT_MODEL = "d3LLM/d3LLM_Dream_Coder"


@dataclass
class GenerationParams:
    max_new_tokens: int = 256
    temperature: float = 0.0
    threshold: float = 0.45
    block_size: int = 32
    block_add_threshold: float = 1.0
    decoded_token_threshold: float = 1.0
    early_stop: bool = True
    alg: str = "entropy_threshold"


@dataclass
class StreamChunk:
    """流式生成的一次中间结果。"""

    trace_steps: List[Dict[str, Any]]
    partial_text: str
    nfe: int
    step_index: int
    elapsed_sec: float
    done: bool = False
    result: Optional["GenerationResult"] = None


@dataclass
class GenerationResult:
    assistant_text: str
    trace_steps: List[Dict[str, Any]] = field(default_factory=list)
    nfe: int = 0
    elapsed_sec: float = 0.0
    num_tokens: int = 0
    tps: float = 0.0
    tpf: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)


class DreamCoderRunner:
    def __init__(self, model_path: str = DEFAULT_MODEL, device: Optional[str] = None):
        self.model_path = model_path
        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> str:
        if self._loaded:
            return f"模型已加载: {self.model_path} @ {self.device}"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        model_config = DreamConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        try:
            model_config._attn_implementation = "sdpa"
        except Exception:
            pass

        self.model = DreamModel.from_pretrained(
            self.model_path,
            config=model_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.model = self.model.to(self.device).eval()

        mixin = DreamGenerationMixin()
        self.model.generate_multi_block = types.MethodType(
            mixin.generate_multi_block, self.model
        )
        self.model._sample_multi_block = types.MethodType(
            mixin._sample_multi_block, self.model
        )
        self.model._sample_multi_block_kv_cache = types.MethodType(
            mixin._sample_multi_block_kv_cache, self.model
        )
        self.model._prepare_inputs = types.MethodType(
            mixin._prepare_inputs, self.model
        )

        self._loaded = True
        return f"✅ 模型加载完成: {self.model_path} | 设备: {self.device} | dtype: bfloat16"

    def _align_max_new_tokens(self, max_new_tokens: int, block_size: int) -> int:
        if max_new_tokens % block_size != 0:
            return (max_new_tokens // block_size) * block_size
        return max_new_tokens

    def _extract_assistant(self, full_ids: torch.Tensor, prompt_len: int) -> str:
        gen_ids = full_ids[prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        eos = self.tokenizer.eos_token or ""
        if eos and eos in text:
            text = text.split(eos)[0]
        return text.strip()

    def _partial_text_from_gen_ids(
        self, gen_token_ids: List[int], mask_token_id: int
    ) -> str:
        ids = [int(t) for t in gen_token_ids if int(t) != mask_token_id]
        if not ids:
            return "▌"
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        eos = self.tokenizer.eos_token or ""
        if eos and eos in text:
            text = text.split(eos)[0]
        return text.strip() or "▌"

    @torch.no_grad()
    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        params: Optional[GenerationParams] = None,
    ) -> Generator[StreamChunk, None, None]:
        if not self._loaded:
            self.load()

        params = params or GenerationParams()
        max_new_tokens = self._align_max_new_tokens(
            params.max_new_tokens, params.block_size
        )
        mask_token_id = int(self.model.config.mask_token_id)

        prompt_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get(
            "attention_mask", torch.ones_like(input_ids)
        ).to(self.device)
        prompt_len = input_ids.shape[1]

        gen_config = DreamGenerationConfig(
            max_length=prompt_len + max_new_tokens,
            mask_token_id=mask_token_id,
            temperature=params.temperature,
            alg=params.alg,
            return_dict_in_generate=True,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        start = time.time()
        trace_steps: List[Dict[str, Any]] = []
        nfe = 0

        for event in sample_multi_block_with_trace_stream(
            self.model,
            input_ids,
            attention_mask,
            gen_config,
            self.tokenizer,
            threshold=params.threshold,
            block_size=params.block_size,
            block_add_threshold=params.block_add_threshold,
            decoded_token_threshold=params.decoded_token_threshold,
            early_stop=params.early_stop,
        ):
            if event["type"] == "step":
                trace_steps = event["trace_steps"]
                nfe = event["nfe"]
                step = trace_steps[-1]
                partial = self._partial_text_from_gen_ids(
                    step["gen_token_ids"], mask_token_id
                )
                yield StreamChunk(
                    trace_steps=list(trace_steps),
                    partial_text=partial,
                    nfe=nfe,
                    step_index=len(trace_steps) - 1,
                    elapsed_sec=time.time() - start,
                    done=False,
                )
            elif event["type"] == "done":
                output = event["output"]
                trace_steps = event["trace_steps"]
                nfe = event["nfe"]
                elapsed = time.time() - start
                assistant_text = self._extract_assistant(output.sequences[0], prompt_len)
                num_tokens = len(
                    self.tokenizer.encode(assistant_text, add_special_tokens=False)
                )
                tps = num_tokens / elapsed if elapsed > 0 else 0.0
                tpf = num_tokens / nfe if nfe > 0 else 0.0
                result = GenerationResult(
                    assistant_text=assistant_text,
                    trace_steps=trace_steps,
                    nfe=nfe,
                    elapsed_sec=elapsed,
                    num_tokens=num_tokens,
                    tps=tps,
                    tpf=tpf,
                    params={
                        "max_new_tokens": max_new_tokens,
                        "temperature": params.temperature,
                        "threshold": params.threshold,
                        "block_size": params.block_size,
                        "block_add_threshold": params.block_add_threshold,
                        "decoded_token_threshold": params.decoded_token_threshold,
                        "early_stop": params.early_stop,
                        "alg": params.alg,
                    },
                )
                yield StreamChunk(
                    trace_steps=list(trace_steps),
                    partial_text=assistant_text,
                    nfe=nfe,
                    step_index=len(trace_steps) - 1,
                    elapsed_sec=elapsed,
                    done=True,
                    result=result,
                )

    @torch.no_grad()
    def generate(
        self,
        messages: List[Dict[str, str]],
        params: Optional[GenerationParams] = None,
    ) -> GenerationResult:
        if not self._loaded:
            self.load()

        params = params or GenerationParams()
        max_new_tokens = self._align_max_new_tokens(
            params.max_new_tokens, params.block_size
        )

        prompt_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get(
            "attention_mask", torch.ones_like(input_ids)
        ).to(self.device)
        prompt_len = input_ids.shape[1]

        gen_config = DreamGenerationConfig(
            max_length=prompt_len + max_new_tokens,
            mask_token_id=self.model.config.mask_token_id,
            temperature=params.temperature,
            alg=params.alg,
            return_dict_in_generate=True,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        start = time.time()
        output, nfe, trace_steps = sample_multi_block_with_trace(
            self.model,
            input_ids,
            attention_mask,
            gen_config,
            self.tokenizer,
            threshold=params.threshold,
            block_size=params.block_size,
            block_add_threshold=params.block_add_threshold,
            decoded_token_threshold=params.decoded_token_threshold,
            early_stop=params.early_stop,
        )
        elapsed = time.time() - start

        assistant_text = self._extract_assistant(output.sequences[0], prompt_len)
        num_tokens = len(
            self.tokenizer.encode(assistant_text, add_special_tokens=False)
        )
        tps = num_tokens / elapsed if elapsed > 0 else 0.0
        tpf = num_tokens / nfe if nfe > 0 else 0.0

        return GenerationResult(
            assistant_text=assistant_text,
            trace_steps=trace_steps,
            nfe=nfe,
            elapsed_sec=elapsed,
            num_tokens=num_tokens,
            tps=tps,
            tpf=tpf,
            params={
                "max_new_tokens": max_new_tokens,
                "temperature": params.temperature,
                "threshold": params.threshold,
                "block_size": params.block_size,
                "block_add_threshold": params.block_add_threshold,
                "decoded_token_threshold": params.decoded_token_threshold,
                "early_stop": params.early_stop,
                "alg": params.alg,
            },
        )


_runner: Optional[DreamCoderRunner] = None


def get_runner(model_path: str = DEFAULT_MODEL) -> DreamCoderRunner:
    global _runner
    if _runner is None or _runner.model_path != model_path:
        _runner = DreamCoderRunner(model_path=model_path)
    return _runner
