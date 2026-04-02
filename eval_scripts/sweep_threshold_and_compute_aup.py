#!/usr/bin/env python
"""
用法示例（在项目根目录 chenx/d3LLM 下运行）:

python eval_scripts/sweep_threshold_and_compute_aup.py gsm8k_cot_zeroshot \
  --max_new_tokens 256 --diffusion_steps 256 \
  --method_name merged_d3LLM_LLaDA_8670 \
  --threshold_init 0 \
  --merged_model /path/to/merged_model \
  --dream_or_ladda ladda

脚本行为:
- 从 threshold=0.0 开始，每次增加 0.1 调用 run_merged_d3llm_llada_eval.sh 或 run_merged_d3llm_dream_eval.sh（由 --dream_or_ladda 决定）
- 记录每个 threshold 得到的 (TPF, Acc)
- 如果当前 Acc < 历史最大 Acc - 5，则停止继续增大 threshold
- 不在脚本内计算 AUP，而是写出 AUP_leaderboard 可直接读取的 YAML 格式数据
"""

import argparse
import os
import re
import subprocess
from typing import List, Tuple
import yaml


def run_single_eval(
    task_name: str,
    max_new_tokens: int,
    diffusion_steps: int,
    threshold: float,
    dream_or_ladda: str,
    merged_model: str,
) -> Tuple[float, float]:
    """
    调用对应的 run_merged_d3llm_*.sh，返回 (TPF, Acc)。
    """
    if dream_or_ladda == "ladda":
        script_path = "eval_scripts/run_merged_d3llm_llada_eval.sh"
    elif dream_or_ladda == "dream":
        script_path = "eval_scripts/run_merged_d3llm_dream_eval.sh"
    else:
        raise ValueError(f"dream_or_ladda 只能是 'dream' 或 'ladda'，当前: {dream_or_ladda}")

    cmd = [
        "bash",
        script_path,
        task_name,
        str(max_new_tokens),
        str(diffusion_steps),
        str(threshold),
        merged_model,
    ]

    proc = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )

    tpf = None
    metric_pairs: List[Tuple[str, float]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        # 运行日志中会多次出现 Tokens per forward，取最后一次
        if "Tokens per forward" in line:
            m_tpf = re.search(r"Tokens per forward:\s*([0-9]+(?:\.[0-9]+)?)", line)
            if m_tpf:
                tpf = float(m_tpf.group(1))
            continue

        # 简要结果中的指标行，形如: "exact_match,none: 0.123456"
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            try:
                metric_pairs.append((key, float(val)))
            except ValueError:
                pass

    # 优先 exact_match,flexible-extract（避免先命中 strict-match）
    acc = None
    for k, v in metric_pairs:
        kl = k.lower().replace(" ", "")
        if "exact_match" in kl and "flexible-extract" in kl and "stderr" not in kl:
            acc = v
            break

    # 其次：任意 exact_match* 且非 stderr
    if acc is None:
        for k, v in metric_pairs:
            kl = k.lower()
            if "exact_match" in kl and "stderr" not in kl:
                acc = v
                break

    # 回退：首个非 stderr 指标
    if acc is None:
        for k, v in metric_pairs:
            if "stderr" not in k.lower():
                acc = v
                break

    if tpf is None or acc is None:
        raise RuntimeError(
            f"无法从脚本输出中解析 TPF/Acc，threshold={threshold}, 输出如下：\n{proc.stdout}"
        )

    return tpf, acc


def to_percent_if_needed(acc: float) -> float:
    """AUP_leaderboard 期望精度是百分制 [0, 100]。若传入 [0,1] 则自动转成百分制。"""
    if 0.0 <= acc <= 1.0:
        return acc * 100.0
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "自动扫描 threshold，并调用 run_merged_d3llm_llada_eval.sh 或 run_merged_d3llm_dream_eval.sh。"
            "将 (TPF,Acc) 写成 AUP_leaderboard 可直接读取的 YAML 数据。"
        )
    )
    parser.add_argument("task_name", help="任务名，例如 gsm8k_cot_zeroshot")
    parser.add_argument(
        "--threshold_init",
        type=float,
        default=0.0,
        help="threshold 的初始值（默认: 0）",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="max_new_tokens (默认: 256)",
    )
    parser.add_argument(
        "--diffusion_steps",
        type=int,
        default=256,
        help="diffusion_steps (默认: 256)",
    )
    parser.add_argument(
        "--method_name",
        type=str,
        default="merged_d3LLM_LLaDA",
        help="写入 YAML 时的方法名键，例如 merged_d3LLM_LLaDA_8670",
    )
    parser.add_argument(
        "--merged_model",
        type=str,
        required=True,
        help="merged_model 的地址（必填）。会作为参数传递给对应的 eval 脚本。",
    )
    parser.add_argument(
        "--dream_or_ladda",
        type=str,
        required=True,
        choices=["dream", "ladda"],
        help="选择 dream 或 ladda（决定调用哪个 run_merged_d3llm_*.sh，且注入对应环境变量）。",
    )
    parser.add_argument(
        "--out_yaml",
        type=str,
        default="AUP_leaderboard/data_custom.yaml",
        help="输出 YAML 路径（默认: AUP_leaderboard/data_custom.yaml）",
    )

    args = parser.parse_args()

    points: List[Tuple[float, float]] = []
    max_acc: float = -1.0

    threshold = float(args.threshold_init)

    print(
        f"从 threshold={threshold:.2f} 开始，每次增加 0.1，直到 Acc 低于当前最大 Acc 5 以上为止；"
        f"模式={args.dream_or_ladda}，merged_model={args.merged_model}"
    )

    while True:
        print(f"\n=== 运行 threshold={threshold:.2f} ===")
        tpf, acc = run_single_eval(
            args.task_name,
            args.max_new_tokens,
            args.diffusion_steps,
            threshold,
            args.dream_or_ladda,
            args.merged_model,
        )
        acc = to_percent_if_needed(acc)
        print(f"threshold={threshold:.2f} -> TPF={tpf:.4f}, Acc={acc:.6f} (%)")
        points.append((tpf, acc))

        if acc > max_acc:
            max_acc = acc

        # 如果当前点明显低于历史最优（差 5 以上），就停止
        if max_acc >= 0.0 and acc < max_acc - 5:
            print(
                f"Acc={acc:.6f} 已经低于当前最大 Acc={max_acc:.6f} 的 5，停止扫阈值。"
            )
            break

        threshold = round(threshold + 0.1, 10)

    print("\n=== 所有点 (TPF, Acc) ===")
    for rho, y in points:
        print(f"rho(TPF)={rho:.4f}, y(Acc%)={y:.6f}")

    points = sorted(points, key=lambda x: x[0])

    # 写出 AUP_leaderboard 可读取的 YAML:
    # <task_name>:
    #   <method_name>:
    #     - [rho, y]
    out_path = args.out_yaml
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    existing = {}
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                existing = loaded

    if args.task_name not in existing or not isinstance(existing[args.task_name], dict):
        existing[args.task_name] = {}

    existing[args.task_name][args.method_name] = [
        [float(rho), float(y)] for rho, y in points
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(existing, f, sort_keys=False, allow_unicode=True)

    print(f"\n结果已写入: {out_path}")
    print(f"任务: {args.task_name}")
    print(f"方法: {args.method_name}")
    print("可在 AUP_leaderboard/main.py 的 DATA_PATHS 中加入该 YAML 进行 AUP 计算。")


if __name__ == "__main__":
    main()

# python eval_scripts/sweep_threshold_and_compute_aup.py gsm8k_cot_zeroshot --max_new_tokens 256 --diffusion_steps 256 --method_name merged_d3LLM_LLaDA_8670 --threshold_init 0 --merged_model /home/u-chenx/chenx/d3LLM/output_model/merged_d3LLM_LLaDA_8670 --dream_or_ladda ladda --out_yaml AUP_leaderboard/data_custom.yaml >run.log 2>&1