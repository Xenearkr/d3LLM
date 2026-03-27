import json
from pprint import pprint
from pathlib import Path

from datasets import load_dataset


CACHE_DIR = "/home/u-shengbf/.cache/huggingface/datasets/Zigeng___d_parallel_dream_distill_data/default/0.0.0/f1bb3bea262cf38c54cbd3a21520595abae1c97c"


def shorten(value, max_len=800):
    """Pretty-print helper: truncate very long strings."""
    if isinstance(value, str) and len(value) > max_len:
        return value[:max_len] + f"\n... <truncated, total {len(value)} chars>"
    return value


def main():
    cache_dir = Path(CACHE_DIR)
    arrow_file = cache_dir / "d_parallel_dream_distill_data-train.arrow"

    if not arrow_file.exists():
        raise FileNotFoundError(f"Arrow file not found: {arrow_file}")

    # 直接从本地 arrow 文件加载
    ds = load_dataset(
        "arrow",
        data_files={"train": str(arrow_file)},
        split="train",
    )

    print("=" * 80)
    print("Dataset loaded successfully")
    print("=" * 80)
    print(f"num_rows     : {len(ds)}")
    print(f"column_names : {ds.column_names}")
    print("features     :")
    pprint(ds.features)

    # 取第一条样本
    idx = 0
    sample = ds[idx]

    print("\n" + "=" * 80)
    print(f"Sample #{idx}")
    print("=" * 80)

    for k, v in sample.items():
        print(f"\n--- {k} ---")
        if isinstance(v, str):
            print(shorten(v))
        else:
            pprint(v)

    # 额外导出成 json，方便你慢慢看
    output_path = cache_dir / "sample_0_readable.json"
    readable_sample = {k: shorten(v, max_len=5000) for k, v in sample.items()}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(readable_sample, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"Saved readable sample to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()