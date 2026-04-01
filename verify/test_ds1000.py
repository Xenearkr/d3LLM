import json
import traceback
from multiprocessing import get_context

from datasets import load_dataset
from tqdm import tqdm


def normalize_solution(code: str) -> str:
    if code is None:
        return ""
    code = str(code).strip()
    if code.startswith("```"):
        code = code.replace("```python", "").replace("```", "").strip()
    return code


def run_test_callable(fn, solution: str):
    try:
        ret = fn(solution)
        if ret is False:
            return False, "returned False"
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def run_official_eval_inproc(problem, solution):
    result = {
        "passed": False,
        "exec_passed": False,
        "string_passed": True,
        "error": None,
    }

    try:
        local_ns = {}
        exec(problem["code_context"], local_ns, local_ns)

        test_execution = local_ns.get("test_execution")
        test_string = local_ns.get("test_string")

        if test_execution is None:
            raise RuntimeError("code_context 中未找到 test_execution")

        exec_ok, exec_err = run_test_callable(test_execution, solution)
        result["exec_passed"] = exec_ok

        string_ok = True
        string_err = None
        if callable(test_string):
            string_ok, string_err = run_test_callable(test_string, solution)
        result["string_passed"] = string_ok

        result["passed"] = exec_ok and string_ok

        errs = []
        if exec_err:
            errs.append(f"exec: {exec_err}")
        if string_err:
            errs.append(f"string: {string_err}")
        if errs:
            result["error"] = " | ".join(errs)

        return result

    except Exception:
        result["error"] = traceback.format_exc(limit=5)
        return result


def _worker(problem, solution, q):
    q.put(run_official_eval_inproc(problem, solution))


def eval_one(problem, solution, timeout=90):
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


def main():
    traj = list(load_dataset("XenonGas/dream_coder_trajectory_ds1000")["train"])
    ds1000 = list(load_dataset("xlangai/DS-1000")["test"])

    assert len(traj) == len(ds1000) == 1000

    # 先做 sanity check
    print("traj[0] keys:", traj[0].keys())
    print("ds1000[0] keys:", ds1000[0].keys())
    print("traj[0]['idx'] =", traj[0]["idx"])
    print("traj[0]['question'][:200] =", traj[0]["question"][:200])
    print("ds1000[0]['prompt'][:200] =", ds1000[0]["prompt"][:200])

    results = []
    official_correct = 0
    label_match = 0

    for i, row in enumerate(tqdm(traj, desc="Verifying")):
        problem = ds1000[i]
        solution = normalize_solution(row["llm_answer"])

        eval_result = eval_one(problem, solution, timeout=90)

        official_passed = bool(eval_result["passed"])
        dataset_label = row.get("is_correct", None)
        match = (official_passed == dataset_label) if dataset_label is not None else None

        if official_passed:
            official_correct += 1
        if match is True:
            label_match += 1

        results.append({
            "order_id": i,
            "traj_idx": row["idx"],
            "official_passed": official_passed,
            "official_exec_passed": eval_result["exec_passed"],
            "official_string_passed": eval_result["string_passed"],
            "dataset_is_correct": dataset_label,
            "match_dataset_label": match,
            "error": eval_result["error"],
            "question": row["question"],
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
        "verified_accuracy": official_correct / len(traj),
        "label_match_count": label_match,
        "label_match_rate": label_match / len(traj),
    }

    with open("ds1000_teacher_verify_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()