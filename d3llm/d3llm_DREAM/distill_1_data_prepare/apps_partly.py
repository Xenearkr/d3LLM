import os
import json
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from datasets import load_dataset

# Reuse Dream-Coder generation + code parsing helpers
from d3llm.d3llm_DREAM.distill_1_data_prepare.ds1000_partly import (
    strip_trailing_endoftext,
    extract_assistant_response,
    extract_code_from_generation,
    generate_teacher_model_trajectory,
)


def _load_existing_records(path: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Load existing records from JSON array (legacy) or JSONL (preferred)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return [], None

    with open(path, "r", encoding="utf-8") as f:
        head = ""
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                head = ch
                break

    if head == "[":
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
        records = data if isinstance(data, list) else []
        return records, "json_array"

    # JSONL
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records, "jsonl"


def _select_nonempty_solution(solutions: Any) -> str:
    """APPS solutions is a JSON string that decodes to a list of code strings."""
    if not solutions:
        return ""
    if isinstance(solutions, str):
        # Defensive: sometimes already parsed
        return solutions.strip()
    if isinstance(solutions, list):
        for s in solutions:
            if isinstance(s, str) and s.strip():
                return s
    return ""


def _build_apps_prompt(
    question: str,
    input_output: Optional[Dict[str, Any]],
    starter_code: str,
    max_io_examples: int,
    max_prompt_chars: int,
) -> str:
    parts: List[str] = [question.strip()]

    if input_output:
        inputs = input_output.get("inputs", []) or []
        outputs = input_output.get("outputs", []) or []
        n = min(len(inputs), len(outputs), max_io_examples)
        if n > 0:
            io_chunks = []
            for i in range(n):
                inp = inputs[i]
                out = outputs[i]
                io_chunks.append(f"Example {i} Input:\n{inp}\nExample {i} Output:\n{out}")
            parts.append("\n\n".join(io_chunks))

    if starter_code and isinstance(starter_code, str) and starter_code.strip():
        parts.append("Starter code:\n" + starter_code.strip())

    parts.append(
        "Please output only Python code, do not include explanations or markdown code fences."
    )

    prompt = "\n\n".join(parts).strip()
    if len(prompt) > max_prompt_chars:
        prompt = prompt[:max_prompt_chars]
    return prompt


def main(
    start_idx: int,
    end_idx: int,
    steps: int = 512,
    gen_length: int = 512,
    block_length: int = 32,
    output_file: str = "trajectory_part.jsonl",
    max_data_num: int = -1,
    trajectory_one_step: bool = False,
    stride: int = 1,
    dataset_split: str = "train",
    max_io_examples: int = 2,
    max_prompt_chars: int = 8000,
):
    device = "cuda"

    model_path = "Dream-org/Dream-Coder-v0-Instruct-7B"
    teacher_model = AutoModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # codeparrot/apps needs HF "trust_remote_code" to run its custom dataset loader.
    dataset = load_dataset("codeparrot/apps", split=dataset_split, trust_remote_code=True)

    # Apply max_data_num against global prefix [0, max_data_num)
    if max_data_num > 0:
        end_idx = min(end_idx, max_data_num)

    existing_records, existing_format = _load_existing_records(output_file)
    processed_idx = {int(r["idx"]) for r in existing_records if "idx" in r}
    print(f"[{os.path.basename(output_file)}] Found {len(processed_idx)} existing idx")

    # Convert legacy json array to JSONL for append
    if existing_format == "json_array":
        with open(output_file, "w", encoding="utf-8") as f:
            for rec in existing_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        processed_idx = {int(r["idx"]) for r in existing_records if "idx" in r}

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    out_f = open(output_file, "a", encoding="utf-8")

    new_count = 0
    try:
        for idx in tqdm(
            range(start_idx, min(end_idx, len(dataset)), max(1, stride)),
            desc="Generating trajectories",
        ):
            if idx in processed_idx:
                continue

            sample = dataset[idx]
            question = sample.get("question", "")
            starter_code = sample.get("starter_code", "") or ""

            # Parse solutions/input_output (stored as JSON text)
            solutions_raw = sample.get("solutions", "")
            input_output_raw = sample.get("input_output", "")

            try:
                solutions = json.loads(solutions_raw) if isinstance(solutions_raw, str) else solutions_raw
            except Exception:
                solutions = []

            try:
                input_output = (
                    json.loads(input_output_raw)
                    if isinstance(input_output_raw, str)
                    else input_output_raw
                )
            except Exception:
                input_output = None

            ground_truth_code = _select_nonempty_solution(solutions)

            user_prompt = _build_apps_prompt(
                question=question,
                input_output=input_output,
                starter_code=starter_code,
                max_io_examples=max_io_examples,
                max_prompt_chars=max_prompt_chars,
            )

            messages = [{"role": "user", "content": user_prompt}]
            inputs = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            )
            input_ids = inputs.input_ids.to(device=device)
            attention_mask = inputs.attention_mask.to(device=device)
            prompt_ids = input_ids[0].cpu().tolist()

            # Teacher generation (threshold=-inf keeps trajectory in "one token per step")
            final_output, trajectory, nfe = generate_teacher_model_trajectory(
                teacher_model,
                tokenizer,
                input_ids,
                attention_mask=attention_mask,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=0.0,
                threshold=-float("inf"),
                trajectory_one_step=trajectory_one_step,
            )

            generated_text_full = tokenizer.decode(
                final_output[0],
                skip_special_tokens=False,
            )
            generated_text_full = strip_trailing_endoftext(generated_text_full)
            assistant_output = extract_assistant_response(generated_text_full)
            generated_code = extract_code_from_generation(assistant_output)
            llm_answer = generated_code if generated_code.strip() else assistant_output

            if trajectory_one_step:
                processed_trajectory = trajectory[0].cpu().tolist()
            else:
                processed_trajectory = [traj[0].cpu().tolist() for traj in trajectory]

            result_obj = {
                "idx": idx,
                "question": user_prompt,
                "prompt_ids": prompt_ids,
                "trajectory": processed_trajectory,
                "final_output": final_output[0].cpu().tolist(),
                "generated_text": generated_text_full,
                "llm_answer": llm_answer,
                "gt_answer": ground_truth_code,
                "is_correct": True,
                "nfe": nfe,
            }

            out_f.write(json.dumps(result_obj, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())
            processed_idx.add(idx)
            new_count += 1
    finally:
        out_f.close()

    print(f"Appended {new_count} new trajectories to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, required=True)
    parser.add_argument("--end_idx", type=int, required=True)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--gen_length", type=int, default=512)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--output_file", type=str, default="trajectory_part.jsonl")
    parser.add_argument("--max_data_num", type=int, default=-1)
    parser.add_argument("--trajectory_one_step", action="store_true")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--max_io_examples", type=int, default=2)
    parser.add_argument("--max_prompt_chars", type=int, default=8000)

    args = parser.parse_args()
    main(
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        output_file=args.output_file,
        max_data_num=args.max_data_num,
        trajectory_one_step=args.trajectory_one_step,
        stride=args.stride,
        dataset_split=args.dataset_split,
        max_io_examples=args.max_io_examples,
        max_prompt_chars=args.max_prompt_chars,
    )

