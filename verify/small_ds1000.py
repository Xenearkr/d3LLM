from datasets import load_dataset
import json

traj = load_dataset("XenonGas/dream_coder_trajectory_ds1000")["train"]

for i in range(5):
    row = traj[i]
    print("=" * 80)
    print("keys:", row.keys())
    print("llm_answer type:", type(row["llm_answer"]))
    print("llm_answer preview:")
    print(str(row["llm_answer"])[:1200])