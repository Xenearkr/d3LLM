"""
带解码轨迹记录的多块扩散生成（基于 d3llm_DREAM._sample_multi_block）。
支持逐步 yield，供 Gradio 流式界面使用。
"""
from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
import torch.nn.functional as F

from d3llm.d3llm_DREAM.d3llm_dream_generate_util import (
    DreamGenerationConfig,
    DreamModelOutput,
    handle_early_stop,
)


def format_gen_region_display(
    tokenizer,
    gen_token_ids: List[int],
    mask_token_id: int,
    *,
    total_slots: Optional[int] = None,
) -> str:
    """逐 token 拼接固定长度全文；mask 位置显示 <mask>，避免整段 decode 吞掉尾部未填位置。"""
    ids = [int(t) for t in gen_token_ids]
    if total_slots is not None and total_slots > len(ids):
        ids = ids + [int(mask_token_id)] * (total_slots - len(ids))
    parts: List[str] = []
    for tid in ids:
        if int(tid) == mask_token_id:
            parts.append("<mask>")
        else:
            parts.append(tokenizer.decode([int(tid)], skip_special_tokens=False))
    return "".join(parts)


def _block_summary(block_states: Dict[int, dict]) -> List[dict]:
    rows = []
    for bid in sorted(block_states.keys()):
        bs = block_states[bid]
        total = max(bs["total_masks"], 1)
        progress = 1.0 - bs["mask_count"] / total
        rows.append(
            {
                "block_id": bid,
                "start": bs["start"],
                "end": bs["end"],
                "mask_remaining": int(bs["mask_count"]),
                "progress": round(progress, 4),
                "is_complete": bool(bs["is_complete"]),
            }
        )
    return rows


def _append_step_record(
    trace_steps: List[Dict[str, Any]],
    *,
    step_no: int,
    decoded_this_step: List[Dict[str, Any]],
    masks_remaining: int,
    active_end: int,
    block_states: Dict[int, dict],
    gen_slice: List[int],
    gen_region_confidences: List[Optional[float]],
    snapshot_display: str,
) -> Dict[str, Any]:
    record = {
        "step": step_no,
        "num_decoded": len(decoded_this_step),
        "masks_remaining": masks_remaining,
        "active_end": int(active_end),
        "decoded_tokens": decoded_this_step,
        "blocks": _block_summary(block_states),
        "sequence_preview": snapshot_display[:500],
        "gen_token_ids": list(gen_slice),
        "gen_confidences": list(gen_region_confidences),
    }
    trace_steps.append(record)
    return record


@torch.no_grad()
def sample_multi_block_with_trace_stream(
    model,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor],
    generation_config: DreamGenerationConfig,
    tokenizer,
    threshold: float = 0.45,
    block_size: int = 32,
    block_add_threshold: float = 1.0,
    decoded_token_threshold: float = 1.0,
    early_stop: bool = False,
) -> Generator[Dict[str, Any], None, None]:
    """每完成一个解码步 yield 一次；结束时 yield type=done。"""
    max_length = generation_config.max_length
    mask_token_id = generation_config.mask_token_id
    temperature = generation_config.temperature
    alg = generation_config.alg
    eos_token_id = generation_config.eos_token_id if early_stop else None

    max_new_tokens = max_length - input_ids.shape[1]
    prompt_length = input_ids.shape[1]
    x = F.pad(input_ids, (0, max_new_tokens), value=mask_token_id)

    if attention_mask is not None and torch.any(attention_mask == 0.0):
        attention_mask_padded = F.pad(attention_mask, (0, max_new_tokens), value=1.0)
        attn_mask_4d = torch.logical_and(
            attention_mask_padded.unsqueeze(1).unsqueeze(-2),
            attention_mask_padded.unsqueeze(1).unsqueeze(-1),
        )
    else:
        attn_mask_4d = "full"

    block_states = {
        0: {
            "start": 0,
            "end": input_ids.shape[1],
            "mask_count": 0,
            "total_masks": input_ids.shape[1],
            "is_complete": True,
        }
    }

    num_blocks = max_new_tokens // block_size
    next_block_id = 1
    if next_block_id <= num_blocks:
        block_start = input_ids.shape[1] + (next_block_id - 1) * block_size
        block_end = min(block_start + block_size, input_ids.shape[1] + max_new_tokens)
        should_activate = 1.0 >= decoded_token_threshold
        block_states[next_block_id] = {
            "start": block_start,
            "end": block_end,
            "mask_count": block_end - block_start,
            "total_masks": block_end - block_start,
            "is_complete": should_activate,
        }
        next_block_id += 1

    nfe = 0
    has_eos = False
    first_eos_abs = None
    trace_steps: List[Dict[str, Any]] = []
    gen_region_confidences: List[Optional[float]] = [None] * max_new_tokens

    gen_region = x[0, prompt_length : prompt_length + max_new_tokens].tolist()
    snapshot_all_mask = format_gen_region_display(
        tokenizer, gen_region, mask_token_id, total_slots=max_new_tokens
    )
    step0 = _append_step_record(
        trace_steps,
        step_no=0,
        decoded_this_step=[],
        masks_remaining=int((x[0, prompt_length:] == mask_token_id).sum().item()),
        active_end=int(prompt_length),
        block_states=block_states,
        gen_slice=gen_region,
        gen_region_confidences=gen_region_confidences,
        snapshot_display=snapshot_all_mask,
    )
    yield {"type": "step", "step": step0, "trace_steps": list(trace_steps), "nfe": 0}

    while True:
        mask_index = x == mask_token_id
        total_masks = mask_index[:, prompt_length:].sum()

        if total_masks == 0 and next_block_id > num_blocks:
            break

        nfe += 1

        if early_stop and eos_token_id is not None:
            has_eos, current_eos_pos = handle_early_stop(
                x,
                block_states,
                eos_token_id,
                prompt_length,
                mask_token_id=mask_token_id,
                debug=False,
            )
            if has_eos:
                first_eos_abs = current_eos_pos
            if has_eos:
                mask_index_updated = x == mask_token_id
                for bid in sorted(block_states.keys()):
                    if bid > 0:
                        start, end = block_states[bid]["start"], block_states[bid]["end"]
                        actual_mask_count = mask_index_updated[:, start:end].sum().item()
                        if actual_mask_count != block_states[bid]["mask_count"]:
                            block_states[bid]["mask_count"] = actual_mask_count
                            if actual_mask_count == 0:
                                block_states[bid]["is_complete"] = True

                while next_block_id <= num_blocks:
                    block_start = prompt_length + (next_block_id - 1) * block_size
                    block_end = min(block_start + block_size, prompt_length + max_new_tokens)
                    if block_start > first_eos_abs:
                        block_states[next_block_id] = {
                            "start": block_start,
                            "end": block_end,
                            "mask_count": 0,
                            "total_masks": block_end - block_start,
                            "is_complete": True,
                        }
                        next_block_id += 1
                    else:
                        break

                total_masks = mask_index_updated[:, prompt_length:].sum()
                if total_masks == 0:
                    break

        for bid in sorted(block_states.keys()):
            if bid > 0 and not block_states[bid]["is_complete"]:
                prev_progress = (
                    1 - block_states[bid - 1]["mask_count"] / block_states[bid - 1]["total_masks"]
                )
                if prev_progress >= decoded_token_threshold:
                    block_states[bid]["is_complete"] = True

        if next_block_id <= num_blocks and not has_eos:
            last_bid = max(block_states.keys())
            if last_bid > 0:
                last_progress = (
                    1 - block_states[last_bid]["mask_count"] / block_states[last_bid]["total_masks"]
                )
                should_add_block = (last_progress >= block_add_threshold) or (
                    block_states[last_bid]["mask_count"] == 0
                )
                if should_add_block:
                    block_start = input_ids.shape[1] + (next_block_id - 1) * block_size
                    block_end = min(block_start + block_size, input_ids.shape[1] + max_new_tokens)
                    if block_end > block_start:
                        actual_mask_count = (x[:, block_start:block_end] == mask_token_id).sum().item()
                        prev_bid = next_block_id - 1
                        prev_progress = (
                            1
                            - block_states[prev_bid]["mask_count"]
                            / block_states[prev_bid]["total_masks"]
                        )
                        should_activate = prev_progress >= decoded_token_threshold
                        block_states[next_block_id] = {
                            "start": block_start,
                            "end": block_end,
                            "mask_count": actual_mask_count,
                            "total_masks": block_end - block_start,
                            "is_complete": should_activate,
                        }
                        next_block_id += 1

        rightmost_active_bid = 0
        for bid in sorted(block_states.keys()):
            if block_states[bid]["is_complete"] or block_states[bid]["mask_count"] > 0:
                rightmost_active_bid = bid

        if rightmost_active_bid == 0:
            break

        active_end = block_states[rightmost_active_bid]["end"]
        model_output = model(x, attn_mask_4d, None)
        logits = model_output.logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        mask_index_for_decode = mask_index.clone()
        mask_index_for_decode[:, active_end:] = 0

        decoded_this_step: List[Dict[str, Any]] = []

        if alg == "entropy_threshold":
            p = F.softmax(logits.to(torch.float64), dim=-1)
            entropy = -torch.sum(p * torch.log(p + 1e-12), dim=-1)
            confidence = p.max(dim=-1).values

            x0 = torch.argmax(logits, dim=-1)
            if temperature > 0:
                p_temp = F.softmax(logits / temperature, dim=-1)
                x0 = torch.multinomial(p_temp.view(-1, p_temp.shape[-1]), 1).view(x.shape)

            transfer_index = (entropy < threshold) & mask_index_for_decode

            first_fully_activated_bid = None
            for bid in sorted(block_states.keys()):
                if bid > 0 and block_states[bid]["is_complete"] and block_states[bid]["mask_count"] > 0:
                    first_fully_activated_bid = bid
                    break

            forced_decode = False
            if first_fully_activated_bid is not None:
                start, end = block_states[first_fully_activated_bid]["start"], block_states[
                    first_fully_activated_bid
                ]["end"]
                block_transfer = transfer_index[:, start:end]
                if not block_transfer.any():
                    block_mask = mask_index_for_decode[:, start:end]
                    block_entropy = entropy[:, start:end]
                    block_entropy = torch.where(block_mask, block_entropy, torch.inf)
                    best_idx = block_entropy[0].argmin()
                    transfer_index[0, start + best_idx] = True
                    forced_decode = True

            decode_positions = transfer_index[0].nonzero(as_tuple=False).flatten().tolist()
            for pos in decode_positions:
                tid = int(x0[0, pos].item())
                ent = float(entropy[0, pos].item())
                conf = float(confidence[0, pos].item())
                token_str = tokenizer.decode([tid], skip_special_tokens=False)
                decoded_this_step.append(
                    {
                        "position": int(pos),
                        "token_id": tid,
                        "token": token_str,
                        "entropy": round(ent, 6),
                        "confidence": round(conf, 6),
                        "forced": forced_decode and pos == decode_positions[-1],
                    }
                )
                rel_i = int(pos) - prompt_length
                if 0 <= rel_i < max_new_tokens:
                    gen_region_confidences[rel_i] = conf

            x[transfer_index] = x0[transfer_index]

            for bid in sorted(block_states.keys()):
                if bid > 0 and block_states[bid]["mask_count"] > 0:
                    start, end = block_states[bid]["start"], block_states[bid]["end"]
                    block_decoded = transfer_index[:, start:end].sum().item()
                    if block_decoded > 0:
                        block_states[bid]["mask_count"] -= block_decoded

        gen_slice = x[0, prompt_length : prompt_length + max_new_tokens].tolist()
        snapshot_display = format_gen_region_display(
            tokenizer, gen_slice, mask_token_id, total_slots=max_new_tokens
        )

        step_rec = _append_step_record(
            trace_steps,
            step_no=nfe,
            decoded_this_step=decoded_this_step,
            masks_remaining=int((x[0, prompt_length:] == mask_token_id).sum().item()),
            active_end=int(active_end),
            block_states=block_states,
            gen_slice=gen_slice,
            gen_region_confidences=gen_region_confidences,
            snapshot_display=snapshot_display,
        )
        yield {"type": "step", "step": step_rec, "trace_steps": list(trace_steps), "nfe": nfe}

        if nfe > 10000:
            break

    yield {
        "type": "done",
        "output": DreamModelOutput(sequences=x),
        "nfe": nfe,
        "trace_steps": trace_steps,
        "prompt_length": prompt_length,
    }


@torch.no_grad()
def sample_multi_block_with_trace(
    model,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor],
    generation_config: DreamGenerationConfig,
    tokenizer,
    threshold: float = 0.45,
    block_size: int = 32,
    block_add_threshold: float = 1.0,
    decoded_token_threshold: float = 1.0,
    early_stop: bool = False,
) -> Tuple[DreamModelOutput, int, List[Dict[str, Any]]]:
    output, nfe, trace_steps = None, 0, []
    for event in sample_multi_block_with_trace_stream(
        model,
        input_ids,
        attention_mask,
        generation_config,
        tokenizer,
        threshold=threshold,
        block_size=block_size,
        block_add_threshold=block_add_threshold,
        decoded_token_threshold=decoded_token_threshold,
        early_stop=early_stop,
    ):
        if event["type"] == "done":
            output = event["output"]
            nfe = event["nfe"]
            trace_steps = event["trace_steps"]
    return output, nfe, trace_steps
