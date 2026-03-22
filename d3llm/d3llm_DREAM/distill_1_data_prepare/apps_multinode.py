import subprocess
import json
import os
import time
import argparse
from typing import Any, Dict, List

from datasets import Dataset, load_dataset


def _load_part_records(p: str) -> List[Dict[str, Any]]:
    """Load JSON array (legacy) or JSONL (preferred) into list[dict]."""
    with open(p, "r", encoding="utf-8") as f:
        head = ""
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                head = ch
                break

    if head == "[":
        with open(p, "r", encoding="utf-8") as f2:
            try:
                data_arr = json.load(f2)
            except json.JSONDecodeError:
                data_arr = []
        return data_arr if isinstance(data_arr, list) else []

    records: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f2:
        for line in f2:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _normalize_trajectory_for_arrow(traj: Any) -> List[List[int]]:
    """Unify trajectory nesting so PyArrow can store one schema (2 levels: step → tokens).

    JSONL may mix per-step shapes: ``list[int]`` (tokens) or ``list[list[int]]``
    (batch=1 wrapper). Both are collapsed to ``list[int]`` per step so the Arrow
    column is ``Sequence(Sequence(int64))``, matching DS1000-style datasets.
    """
    if traj is None or not isinstance(traj, list):
        return []
    out: List[List[int]] = []
    for step in traj:
        if not isinstance(step, list):
            continue
        if len(step) == 0:
            out.append([])
        elif isinstance(step[0], int):
            out.append(step)
        elif isinstance(step[0], list):
            # Drop singleton batch dimension: [[tok, ...]] → [tok, ...]
            out.append(step[0])
        else:
            out.append([])
    return out


def main(
    num_gpus: int = 24,
    steps: int = 512,
    gen_length: int = 512,
    block_length: int = 32,
    output_dir: str = "trajectory_output_apps",
    max_data_num: int = -1,
    trajectory_one_step: bool = False,
    dataset_split: str = "train",
    max_prompt_chars: int = 8000,
    max_io_examples: int = 2,
):
    slurm_procid = int(os.environ.get("SLURM_PROCID", "0"))
    slurm_localid = int(os.environ.get("SLURM_LOCALID", "0"))
    slurm_ntasks = int(os.environ.get("SLURM_NTASKS", str(num_gpus)))

    print(f"Task {slurm_procid}/{slurm_ntasks}, Local GPU {slurm_localid}")

    dataset_name = "codeparrot/apps"

    # Only rank0 loads dataset size and writes total_size.txt for others.
    if slurm_procid == 0:
        # codeparrot/apps needs HF "trust_remote_code" to run its custom dataset loader.
        dataset = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)
        total_size = len(dataset)
        if max_data_num > 0:
            total_size = min(total_size, max_data_num)

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "total_size.txt"), "w", encoding="utf-8") as f:
            f.write(str(total_size))

        print(f"Total dataset size: {total_size} from {dataset_name}[{dataset_split}]")

    total_size_file = os.path.join(output_dir, "total_size.txt")
    while not os.path.exists(total_size_file):
        time.sleep(1)

    with open(total_size_file, "r", encoding="utf-8") as f:
        total_size = int(f.read().strip())

    # Strided partition by modulo.
    gpu_id = slurm_procid
    start_idx = gpu_id
    end_idx = total_size

    output_file = os.path.join(output_dir, f"trajectory_part_{gpu_id}.jsonl")
    completion_file = os.path.join(output_dir, f"completed_{gpu_id}.flag")

    # Resume fix: remove stale completion flags from previous interrupted runs.
    if os.path.exists(completion_file):
        os.remove(completion_file)

    cmd = [
        "python",
        "-m",
        "d3llm.d3llm_DREAM.distill_1_data_prepare.apps_partly",
        "--start_idx",
        str(start_idx),
        "--end_idx",
        str(end_idx),
        "--steps",
        str(steps),
        "--gen_length",
        str(gen_length),
        "--block_length",
        str(block_length),
        "--output_file",
        output_file,
        "--max_data_num",
        str(max_data_num),
        "--stride",
        str(num_gpus),
        "--dataset_split",
        dataset_split,
        "--max_prompt_chars",
        str(max_prompt_chars),
        "--max_io_examples",
        str(max_io_examples),
    ]
    if trajectory_one_step:
        cmd.append("--trajectory_one_step")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(slurm_localid)

    print(f"GPU {gpu_id}: Generating idx start={start_idx}, stride={num_gpus}, end<{end_idx}")
    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        print(f"GPU {gpu_id}: Generation failed with return code {result.returncode}")
        raise SystemExit(1)

    print(f"GPU {gpu_id}: Generation completed")

    # Write completion flag for this rank.
    with open(completion_file, "w", encoding="utf-8") as f:
        f.write("done")

    # Only rank0 concatenates results.
    if slurm_procid == 0:
        print("Waiting for all tasks to complete...")
        while True:
            completed = sum(
                1
                for i in range(num_gpus)
                if os.path.exists(os.path.join(output_dir, f"completed_{i}.flag"))
            )
            if completed == num_gpus:
                break
            print(f"Completed: {completed}/{num_gpus}")
            time.sleep(5)

        print("All tasks completed. Concatenating results...")

        parts: List[List[Dict[str, Any]]] = []
        max_len = 0
        for part_rank in range(num_gpus):
            part_file = os.path.join(output_dir, f"trajectory_part_{part_rank}.jsonl")
            if not os.path.exists(part_file) or os.path.getsize(part_file) == 0:
                print(f"Warning: {part_file} not found/empty")
                parts.append([])
                continue
            data = _load_part_records(part_file)
            print(f"Loaded {len(data)} samples from GPU {part_rank}")
            parts.append(data)
            max_len = max(max_len, len(data))

        # Round-robin merge to preserve global idx order without sorting.
        all_data: List[Dict[str, Any]] = []
        seen_idx = set()
        for i in range(max_len):
            for part_rank in range(num_gpus):
                if i >= len(parts[part_rank]):
                    continue
                rec = parts[part_rank][i]
                idx_val = rec.get("idx", None)
                if idx_val is not None:
                    try:
                        idx_int = int(idx_val)
                    except Exception:
                        idx_int = None
                    if idx_int is not None and idx_int in seen_idx:
                        continue
                    if idx_int is not None:
                        seen_idx.add(idx_int)
                all_data.append(rec)

        dataset_dict = {
            "idx": [d.get("idx", None) for d in all_data],
            "question": [d.get("question", "") for d in all_data],
            "prompt_ids": [d.get("prompt_ids", []) for d in all_data],
            "trajectory": [
                _normalize_trajectory_for_arrow(d.get("trajectory", []))
                for d in all_data
            ],
            "final_output": [d.get("final_output", []) for d in all_data],
            "generated_text": [d.get("generated_text", "") for d in all_data],
            "llm_answer": [d.get("llm_answer", "") for d in all_data],
            "gt_answer": [d.get("gt_answer", "") for d in all_data],
            "is_correct": [d.get("is_correct", True) for d in all_data],
            "nfe": [d.get("nfe", 0) for d in all_data],
        }

        final_dataset = Dataset.from_dict(dataset_dict)
        final_dataset.save_to_disk(os.path.join(output_dir, "trajectory_dataset"))
        print(f"Saved complete dataset with {len(all_data)} samples to {output_dir}/trajectory_dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--gen_length", type=int, default=512)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="trajectory_output_apps")
    parser.add_argument("--max_data_num", type=int, default=-1)
    parser.add_argument("--trajectory_one_step", action="store_true")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--max_prompt_chars", type=int, default=8000)
    parser.add_argument("--max_io_examples", type=int, default=2)
    args = parser.parse_args()

    main(
        num_gpus=args.num_gpus,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        output_dir=args.output_dir,
        max_data_num=args.max_data_num,
        trajectory_one_step=args.trajectory_one_step,
        dataset_split=args.dataset_split,
        max_prompt_chars=args.max_prompt_chars,
        max_io_examples=args.max_io_examples,
    )

