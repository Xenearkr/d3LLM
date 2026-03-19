from datasets import load_dataset


def main():
    dataset = load_dataset("xlangai/DS-1000", split="test")

    print("=" * 80)
    print("DS-1000 loaded successfully")
    print(f"Dataset size: {len(dataset)}")
    print(f"Features: {dataset.features}")
    print("=" * 80)

    required_fields = ["prompt", "reference_code", "code_context", "metadata"]

    # 检查字段是否存在
    feature_names = list(dataset.features.keys())
    for field in required_fields:
        if field not in feature_names:
            raise ValueError(f"Missing required field: {field}")

    print("Required fields check: PASSED")
    print("=" * 80)

    # 抽查前 3 条样本
    inspect_num = min(3, len(dataset))
    for i in range(inspect_num):
        sample = dataset[i]
        print(f"[Sample {i}]")
        print(f"metadata: {sample['metadata']}")
        print(f"prompt preview: {sample['prompt'][:300]!r}")
        print(f"reference_code preview: {sample['reference_code'][:300]!r}")
        print(f"code_context preview: {sample['code_context'][:300]!r}")
        print("-" * 80)

    # 做一个更严格的内容检查
    sample0 = dataset[0]
    assert isinstance(sample0["prompt"], str) and len(sample0["prompt"]) > 0
    assert isinstance(sample0["reference_code"], str) and len(sample0["reference_code"]) > 0
    assert isinstance(sample0["code_context"], str) and len(sample0["code_context"]) > 0
    assert isinstance(sample0["metadata"], dict)

    print("Content sanity check: PASSED")
    print("=" * 80)
    print("DS-1000 import verification finished successfully.")


if __name__ == "__main__":
    main()