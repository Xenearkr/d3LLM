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
- 从 threshold=0.0 开始，每次增加 0.1 调用统一入口 run_code_eval.sh（dream/ladda/dream_coder）
- 记录每个 threshold 得到的 (TPF, Acc)
- 如果当前 Acc < 历史最大 Acc - 5，则停止继续增大 threshold
- 不在脚本内计算 AUP，而是写出 AUP_leaderboard 可直接读取的 YAML 格式数据
"""

import argparse
import glob
import os
import re
import subprocess
import sys
from typing import List, Optional, Tuple
import yaml


def run_single_eval(
    task_name: str,
    max_new_tokens: int,
    diffusion_steps: int,
    threshold: float,
    dream_or_ladda: str,
    merged_model: str,
    force_rerun: bool = True,
) -> Tuple[float, float, Optional[float]]:
    """调用统一评测入口，返回 (TPF, base_acc, plus_acc)。"""
    if dream_or_ladda == "ladda":
        cmd = [
            "bash",
            "eval_scripts/run_merged_d3llm_llada_eval.sh",
            task_name,
            str(max_new_tokens),
            str(diffusion_steps),
            str(threshold),
            merged_model,
        ]
    elif dream_or_ladda == "dream":
        cmd = [
            "bash",
            "eval_scripts/run_code_eval.sh",
            "diffllm",
            merged_model,
            task_name,
            str(max_new_tokens),
            str(diffusion_steps),
            str(threshold),
        ]
    elif dream_or_ladda == "dream_coder":
        if force_rerun:
            _clear_evalplus_cache(
                model=merged_model,
                dataset=task_name,
                max_new_tokens=max_new_tokens,
                threshold=threshold,
            )
        cmd = [
            "bash",
            "eval_scripts/run_code_eval.sh",
            "diffllm_coder",
            merged_model,
            task_name,
            str(max_new_tokens),
            str(diffusion_steps),
            str(threshold),
        ]
    else:
        raise ValueError(f"dream_or_ladda 只能是 'dream' / 'ladda' / 'dream_coder'，当前: {dream_or_ladda}")

    # 实时转发子进程输出，避免长时间无日志（尤其是评测进度条阶段）
    merged_lines: List[str] = []
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        merged_lines.append(line)
    return_code = proc.wait()
    merged_output = "".join(merged_lines)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd, output=merged_output)

    tpf = None
    metric_pairs: List[Tuple[str, float]] = []
    plus_acc: Optional[float] = None
    section: Optional[str] = None  # "base" | "plus"
    for line in merged_output.splitlines():
        line = line.strip()
        # 运行日志中会多次出现吞吐指标，取最后一次。
        # dream/ladda 常见: Tokens per forward
        # dream_coder(evalplus) 常见: Overall token per step / Token per step
        m_tpf = re.search(
            r"(?:Tokens per forward|Overall token per step|Token per step):\s*([0-9]+(?:\.[0-9]+)?)",
            line,
            flags=re.IGNORECASE,
        )
        if m_tpf:
            tpf = float(m_tpf.group(1))
            continue

        # evalplus 输出段落标题，用于区分 base / plus 的 pass@1
        ll = line.lower()
        if "(base tests)" in ll:
            section = "base"
            continue
        if "(base + extra tests)" in ll:
            section = "plus"
            continue

        # evalplus 指标行如: pass@1: 0.744
        m_pass = re.search(r"pass@1:\s*([0-9]+(?:\.[0-9]+)?)", line, flags=re.IGNORECASE)
        if m_pass and section == "plus":
            plus_acc = float(m_pass.group(1))
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

    # dream_coder(evalplus) 常见 pass@1 指标
    if acc is None:
        for k, v in metric_pairs:
            kl = k.lower().replace(" ", "")
            if "pass@1" in kl and "stderr" not in kl:
                acc = v
                break

    # 回退：首个非 stderr 指标
    if acc is None:
        for k, v in metric_pairs:
            if "stderr" not in k.lower():
                acc = v
                break

    if acc is None:
        raise RuntimeError(
            f"无法从脚本输出中解析 Acc，threshold={threshold}\n"
            f"命令: {' '.join(cmd)}\n"
            f"stdout+stderr:\n{merged_output}"
        )
    if tpf is None:
        raise RuntimeError(
            f"无法从脚本输出中解析 TPF（Tokens per forward / token per step），threshold={threshold}\n"
            f"命令: {' '.join(cmd)}\n"
            f"stdout+stderr:\n{merged_output}"
        )

    return tpf, acc, plus_acc


def _evalplus_identifier(model: str, max_new_tokens: int, threshold: float) -> str:
    """Keep naming aligned with evalplus.codegen.run_codegen identifier."""
    identifier = model.strip("./").replace("/", "--") + "_dllm_temp_0.0"
    identifier += f"_len_{max_new_tokens}"
    identifier += "_alg_entropy_threshold"
    # add threshold suffix to avoid different thresholds sharing the same cache key
    identifier += f"_thr_{threshold:.2f}"
    return identifier


def _clear_evalplus_cache(model: str, dataset: str, max_new_tokens: int, threshold: float) -> None:
    """Delete cached evalplus artifacts for a specific model+dataset+threshold run."""
    identifier = _evalplus_identifier(model, max_new_tokens, threshold)
    removed = 0
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # evalplus 的 root 默认是当前工作目录下的 evalplus_results（run_code_eval.sh 在 code_eval 目录执行）
    # 同时兼容旧目录结构，避免路径差异导致“以为清了缓存、实际没清掉”。
    roots = [
        os.path.join(repo_root, "utils", "utils_DreamCoder", "code_eval", "evalplus_results", dataset),
        os.path.join(repo_root, "utils", "utils_DreamCoder", "code_eval", "evalplus", "evalplus_results", dataset),
        os.path.join(os.getcwd(), "evalplus_results", dataset),
    ]
    for root in roots:
        if not os.path.isdir(root):
            continue
        for p in glob.glob(os.path.join(root, f"{identifier}*")):
            try:
                os.remove(p)
                removed += 1
            except IsADirectoryError:
                continue
            except FileNotFoundError:
                continue
    if removed > 0:
        print(f"[force_rerun] 已清理 {removed} 个缓存文件: {identifier}*")
    else:
        print(
            f"[force_rerun] 未找到可清理缓存: {identifier}*；"
            f"已检查目录: {', '.join(roots)}"
        )


def to_percent_if_needed(acc: float) -> float:
    """AUP_leaderboard 期望精度是百分制 [0, 100]。若传入 [0,1] 则自动转成百分制。"""
    if 0.0 <= acc <= 1.0:
        return acc * 100.0
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "自动扫描 threshold，并调用 run_code_eval.sh 的统一入口（dream/ladda/dream_coder）。"
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
        "--threshold_end",
        type=float,
        default=1.0,
        help="threshold 的结束值（默认: 1.0，含端点）",
    )
    parser.add_argument(
        "--threshold_step",
        type=float,
        default=0.1,
        help="threshold 步长（默认: 0.1）",
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
        choices=["dream", "ladda", "dream_coder"],
        help="选择 dream / ladda / dream_coder（决定调用哪条统一评测链路）。",
    )
    parser.add_argument(
        "--out_yaml",
        type=str,
        default="AUP_leaderboard/data_custom.yaml",
        help="输出 YAML 路径（默认: AUP_leaderboard/data_custom.yaml）",
    )
    parser.add_argument(
        "--allow_cache",
        action="store_true",
        help="允许复用历史评测缓存（默认关闭，会强制重跑）。",
    )
    parser.add_argument(
        "--early_stop_drop",
        type=float,
        default=None,
        help=(
            "可选：若设置该值（如 5），当 Acc 低于历史最大值该幅度时提前停止。"
            "默认关闭，保证完整扫完阈值区间。"
        ),
    )

    args = parser.parse_args()

    points: List[Tuple[float, float]] = []
    points_plus: List[Tuple[float, float]] = []
    max_acc: float = -1.0

    threshold = float(args.threshold_init)
    threshold_end = float(args.threshold_end)
    threshold_step = float(args.threshold_step)
    if threshold_step <= 0:
        raise ValueError("--threshold_step 必须 > 0")
    if threshold > threshold_end:
        raise ValueError("--threshold_init 不能大于 --threshold_end")

    print(
        f"从 threshold={threshold:.2f} 开始，每次增加 {threshold_step:.4f}，直到 threshold>{threshold_end:.2f} 为止；"
        f"模式={args.dream_or_ladda}，merged_model={args.merged_model}"
    )
    if args.early_stop_drop is not None:
        print(f"已启用早停：当 Acc 低于历史最大值 {args.early_stop_drop:.4f} 时停止。")

    while threshold <= threshold_end + 1e-12:
        print(f"\n=== 运行 threshold={threshold:.2f} ===")
        tpf, acc, acc_plus = run_single_eval(
            args.task_name,
            args.max_new_tokens,
            args.diffusion_steps,
            threshold,
            args.dream_or_ladda,
            args.merged_model,
            force_rerun=not args.allow_cache,
        )
        acc = to_percent_if_needed(acc)
        print(f"threshold={threshold:.2f} -> TPF={tpf:.4f}, Acc={acc:.6f} (%)")
        points.append((tpf, acc))
        if acc_plus is not None:
            acc_plus = to_percent_if_needed(acc_plus)
            print(f"threshold={threshold:.2f} -> Acc+={acc_plus:.6f} (%)")
            points_plus.append((tpf, acc_plus))

        if acc > max_acc:
            max_acc = acc

        # 可选早停：当前点明显低于历史最优时停止
        if (
            args.early_stop_drop is not None
            and max_acc >= 0.0
            and acc < max_acc - args.early_stop_drop
        ):
            print(
                f"Acc={acc:.6f} 已经低于当前最大 Acc={max_acc:.6f} 的 "
                f"{args.early_stop_drop:.6f}，停止扫阈值。"
            )
            break

        threshold = round(threshold + threshold_step, 10)

    print("\n=== 所有点 (TPF, Acc) ===")
    for rho, y in points:
        print(f"rho(TPF)={rho:.4f}, y(Acc%)={y:.6f}")
    if points_plus:
        print("\n=== 所有点 (TPF, Acc+) ===")
        for rho, y in points_plus:
            print(f"rho(TPF)={rho:.4f}, y(Acc+%)={y:.6f}")

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
    # 对于 evalplus 的代码任务，自动额外写入 humaneval+/mbpp+ 曲线
    if points_plus and args.task_name.lower() in {"humaneval", "mbpp"}:
        plus_task = f"{args.task_name}+"
        if plus_task not in existing or not isinstance(existing[plus_task], dict):
            existing[plus_task] = {}
        existing[plus_task][args.method_name] = [
            [float(rho), float(y)] for rho, y in sorted(points_plus, key=lambda x: x[0])
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