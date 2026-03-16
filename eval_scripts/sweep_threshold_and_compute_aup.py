#!/usr/bin/env python
"""
用法示例（在项目根目录 chenx/d3LLM-new 下运行）:

python eval_scripts/sweep_threshold_and_compute_aup.py gsm8k_cot_zeroshot --max_new_tokens 256 --diffusion_steps 256 --alpha 3.0

脚本行为:
- 从 threshold=0.1 开始，每次增加 0.1 调用 run_merged_d3llm_eval.sh
- 记录每个 threshold 得到的 (TPF, Acc)
- 如果当前 Acc < 历史最大 Acc - 0.05，则停止继续增大 threshold
- 根据所有 (TPF, Acc) 点计算 AUP，并将结果写入:
    eval_scripts/<任务名>_eval.txt
"""

import argparse
import math
import os
import subprocess
from typing import List, Tuple


def run_single_eval(
    task_name: str,
    max_new_tokens: int,
    diffusion_steps: int,
    threshold: float,
) -> Tuple[float, float]:
    """
    调用 run_merged_d3llm_eval.sh，返回 (TPF, Acc)。
    """
    cmd = [
        "bash",
        "eval_scripts/run_merged_d3llm_eval.sh",
        task_name,
        str(max_new_tokens),
        str(diffusion_steps),
        str(threshold),
    ]
    proc = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )

    tpf = None
    acc = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("平均 TPF"):
            # 例如: 平均 TPF (Tokens per forward): 4.73
            try:
                tpf = float(line.split(":")[-1].strip())
            except ValueError:
                pass
        elif line.startswith("准确率"):
            # 例如: 准确率 (exact_match, flexible-extract): 0.799848
            try:
                acc = float(line.split(":")[-1].strip())
            except ValueError:
                pass

    if tpf is None or acc is None:
        raise RuntimeError(
            f"无法从脚本输出中解析 TPF/Acc，threshold={threshold}, 输出如下：\n{proc.stdout}"
        )
    return tpf, acc


def weight_function(y: float, alpha: float, y_max: float) -> float:
    """
    W(y) = min( exp(-alpha * (1 - y / y_max)), 1 )
    """
    if y_max <= 0:
        return 1.0
    val = math.exp(-alpha * (1.0 - y / y_max))
    return min(val, 1.0)


def compute_aup(
    points: List[Tuple[float, float]], alpha: float
) -> float:
    """
    points: [(rho, y)], 其中 rho=TPF, y=Acc
    先按 rho 从小到大排序，然后按公式计算 AUP：

      AUP = ρ1 * y1
            + sum_{i=2}^m (ρ_i - ρ_{i-1}) * ((y_i W(y_i) + y_{i-1} W(y_{i-1})) / 2)
    """
    if not points:
        return 0.0

    # 按 rho 排序
    points = sorted(points, key=lambda x: x[0])
    rhos = [p[0] for p in points]
    ys = [p[1] for p in points]

    y_max = max(ys)

    # ρ1 * y1 （论文原式里是 ρ1 y1）
    aup = rhos[0] * ys[0]

    for i in range(1, len(points)):
        rho_i, y_i = rhos[i], ys[i]
        rho_prev, y_prev = rhos[i - 1], ys[i - 1]
        wi = weight_function(y_i, alpha, y_max)
        w_prev = weight_function(y_prev, alpha, y_max)
        delta_rho = rho_i - rho_prev
        aup += delta_rho * ((y_i * wi + y_prev * w_prev) / 2.0)

    return aup


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "自动扫描 threshold（从 0.1 开始每次+0.1），"
            "调用 run_merged_d3llm_eval.sh，"
            "根据 (TPF,Acc) 计算 AUP，并将结果写入 <任务名>_eval.txt"
        )
    )
    parser.add_argument("task_name", help="任务名，例如 gsm8k_cot_zeroshot")
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
        "--alpha",
        type=float,
        default=3.0,
        help="权重函数中的 penalty factor α (默认: 3.0)",
    )

    args = parser.parse_args()

    points: List[Tuple[float, float]] = []
    max_acc: float = -1.0

    threshold = 0.1

    print("从 threshold=0.1 开始，每次增加 0.1，直到 Acc 低于当前最大 Acc 0.05 以上为止。")

    while True:
        print(f"\n=== 运行 threshold={threshold:.1f} ===")
        tpf, acc = run_single_eval(
            args.task_name,
            args.max_new_tokens,
            args.diffusion_steps,
            threshold,
        )
        print(f"threshold={threshold:.1f} -> TPF={tpf:.4f}, Acc={acc:.6f}")
        points.append((tpf, acc))

        if acc > max_acc:
            max_acc = acc

        # 如果当前点明显低于历史最优（差 0.05 以上），就停止
        if max_acc >= 0.0 and acc < max_acc - 0.05:
            print(
                f"Acc={acc:.6f} 已经低于当前最大 Acc={max_acc:.6f} 的 0.05，停止扫阈值。"
            )
            break

        threshold = round(threshold + 0.1, 10)

    print("\n=== 所有点 (TPF, Acc) ===")
    for rho, y in points:
        print(f"rho(TPF)={rho:.4f}, y(Acc)={y:.6f}")

    aup = compute_aup(points, alpha=args.alpha)
    print(f"\nalpha={args.alpha} 时的 AUP = {aup:.6f}")

    # 将结果写入同目录下 <任务名>_eval.txt
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, f"{args.task_name}_eval.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"task_name: {args.task_name}\n")
        f.write(f"max_new_tokens: {args.max_new_tokens}\n")
        f.write(f"diffusion_steps: {args.diffusion_steps}\n")
        f.write(f"alpha: {args.alpha}\n")
        f.write("\npoints (rho=TPF, y=Acc):\n")
        for rho, y in points:
            f.write(f"rho={rho:.6f}, y={y:.6f}\n")
        f.write(f"\nAUP: {aup:.6f}\n")

    print(f"\n结果已写入: {out_path}")


if __name__ == "__main__":
    main()

# python eval_scripts/sweep_threshold_and_compute_aup.py \
#   gsm8k_cot_zeroshot \
#   --max_new_tokens 256 \
#   --diffusion_steps 256 \
#   --thresholds "0.1,0.2,0.3,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0" \
#   --alpha 3.0