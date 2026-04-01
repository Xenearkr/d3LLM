import json
import traceback
from multiprocessing import get_context
from typing import Dict, Any, List

from datasets import load_dataset
from tqdm import tqdm


TRAJ_DATASET = "XenonGas/dream_coder_trajectory_ds1000"
TRAJ_SPLIT = "train"

DS1000_DATASET = "xlangai/DS-1000"
DS1000_SPLIT = "test"


def normalize_solution(code: str) -> str:
    """
    你的 llm_answer 已经是纯代码字符串。
    这里只做最轻量清洗，避免误伤。
    """
    if code is None:
        return ""
    code = str(code).strip()

    # 去掉偶发 markdown fence
    if code.startswith("```"):
        code = code.replace("```python", "").replace("```", "").strip()

    return code


def run_official_eval_inproc(problem: Dict[str, Any], solution: str) -> Dict[str, Any]:
    """
    在子进程里执行单题评测。
    假设 xlangai/DS-1000 的样本包含 code_context，
    其中定义了 test_execution(solution) 和可选的 test_string(solution)。
    """
    result = {
        "passed": False,
        "exec_passed": False,
        "string_passed": True,
        "error": None,
    }

    try:
        code_context = problem["code_context"]
        local_ns = {}
        exec(code_context, local_ns, local_ns)

        test_execution = local_ns.get("test_execution", None)
        test_string = local_ns.get("test_string", None)

        if test_execution is None:
            raise RuntimeError("code_context 中未找到 test_execution(solution)")

        exec_ok = bool(test_execution(solution))
        result["exec_passed"] = exec_ok

        string_ok = True
        if callable(test_string):
            string_ok = bool(test_string(solution))
        result["string_passed"] = string_ok

        result["passed"] = exec_ok and string_ok
        return result

    except Exception:
        result["error"] = traceback.format_exc(limit=5)
        return result


def _worker(problem: Dict[str, Any], solution: str, q):
    q.put(run_official_eval_inproc(problem, solution))


def eval_one(problem: Dict[str, Any], solution: str, timeout: int = 90) -> Dict[str, Any]:
    ctx = get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(problem, solution, q))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return {
            "passed": False,
            "exec_passed": False,
            "string_passed": False,
            "error": f"Timeout after {timeout}s",
        }

    if not q.empty():
        return q.get()

    return {
        "passed": False,
        "exec_passed": False,
        "string_passed": False,
        "error": "Worker exited without returning result",
    }


def build_ds1000_index(ds_rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    优先按 idx 对齐；如果官方集没有 idx，就退化为顺序对齐。
    """
    if "idx" in ds_rows[0]:
        return {int(r["idx"]): r for r in ds_rows}
    return {i: r for i, r in enumerate(ds_rows)}


def main():
    traj = list(load_dataset(TRAJ_DATASET)[TRAJ_SPLIT])
    ds1000 = list(load_dataset(DS1000_DATASET)[DS1000_SPLIT])

    print(f"Loaded trajectory dataset: {len(traj)}")
    print(f"Loaded DS-1000 official simplified dataset: {len(ds1000)}")

    ds_map = build_ds1000_index(ds1000)

    results = []
    official_correct = 0
    label_match = 0
    skipped = 0

    for row in tqdm(traj, desc="Verifying"):
        idx = int(row["idx"])
        if idx not in ds_map:
            results.append({
                "idx": idx,
                "official_passed": False,
                "official_exec_passed": False,
                "official_string_passed": False,
                "dataset_is_correct": row.get("is_correct"),
                "match_dataset_label": False,
                "error": f"idx {idx} not found in official DS-1000 split",
                "llm_answer": row.get("llm_answer", ""),
            })
            skipped += 1
            continue

        problem = ds_map[idx]
        solution = normalize_solution(row.get("llm_answer", ""))

        eval_result = eval_one(problem, solution, timeout=90)

        official_passed = bool(eval_result["passed"])
        dataset_label = row.get("is_correct", None)
        match = (dataset_label == official_passed) if dataset_label is not None else None

        if official_passed:
            official_correct += 1
        if match is True:
            label_match += 1

        results.append({
            "idx": idx,
            "official_passed": official_passed,
            "official_exec_passed": eval_result["exec_passed"],
            "official_string_passed": eval_result["string_passed"],
            "dataset_is_correct": dataset_label,
            "match_dataset_label": match,
            "error": eval_result["error"],
            "nfe": row.get("nfe"),
            "question": row.get("question"),
            "llm_answer": solution,
            "gt_answer": row.get("gt_answer"),
        })

    with open("ds1000_teacher_verify.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "num_traj_rows": len(traj),
        "num_official_rows": len(ds1000),
        "verified_passed": official_correct,
        "verified_accuracy": official_correct / len(traj) if traj else 0.0,
        "label_match_count": label_match,
        "label_match_rate": label_match / len(traj) if traj else 0.0,
        "skipped": skipped,
    }

    with open("ds1000_teacher_verify_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()