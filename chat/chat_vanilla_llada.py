"""
原生 LLaDA 对话脚本（与 chat_d3llm_llada.py 相同交互方式）。
使用模型：GSAI-ML/LLaDA-8B-Instruct（与 LLaDA-main 一致）。
若遇网络/证书问题：export HF_ENDPOINT=https://hf-mirror.com 或 HF_DISABLE_SSL_VERIFY=1
"""
import os
if os.environ.get("HF_DISABLE_SSL_VERIFY", "").lower() in ("1", "true", "yes"):
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    import requests
    _orig_send = requests.Session.send
    def _send_no_verify(self, request, **kwargs):
        kwargs["verify"] = False
        return _orig_send(self, request, **kwargs)
    requests.Session.send = _send_no_verify

import sys
from pathlib import Path
import time
import torch
import tqdm
from transformers import AutoModel, AutoTokenizer

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "chat"))

from vanilla_llada_generate import generate as llada_generate

# 原生 LLaDA 模型（与 LLaDA-main 一致）
MODEL_PATH = "GSAI-ML/LLaDA-8B-Instruct"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Loading vanilla LLaDA model...")
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = model.to(device).eval()

# 生成参数（与 LLaDA-main/chat.py 一致，原生无 early_stop）
GEN_LENGTH = 128
STEPS = 128
BLOCK_LENGTH = 32
gen_params = {
    "steps": STEPS,
    "gen_length": GEN_LENGTH,
    "block_length": BLOCK_LENGTH,
    "temperature": 0.0,
    "cfg_scale": 0.0,
    "remasking": "low_confidence",
}

print("\n" + "=" * 80)
print("Vanilla LLaDA Chat (GSAI-ML/LLaDA-8B-Instruct). Type 'quit' or 'exit' to end.")
print("=" * 80)

# Warmup
test_questions_path = Path(__file__).parent.parent / "utils" / "serve" / "test_question.txt"
test_questions = []
try:
    with open(test_questions_path, "r") as f:
        content = f.read()
        test_questions = [q.strip() for q in content.split("\n\n") if q.strip()]
except Exception as e:
    print(f"Warning: Could not load test questions: {e}. Using fallback warmup.")
    test_questions = ["Write a hello world program in Python."] * 10

with torch.no_grad():
    for i in tqdm.tqdm(range(min(10, len(test_questions))), desc="Warming up model"):
        prompt_text = test_questions[i % len(test_questions)]
        messages = [{"role": "user", "content": prompt_text}]
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt_text)["input_ids"]
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
        _ = llada_generate(model, input_ids, **gen_params)
print("Warmup complete.\n")

while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break
    if not user_input:
        continue

    messages = [{"role": "user", "content": user_input}]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt_text)["input_ids"]
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    start_time = time.time()
    with torch.no_grad():
        out = llada_generate(model, input_ids, **gen_params)
    end_time = time.time()

    full_response = tokenizer.batch_decode(out[:, input_ids.shape[1] :], skip_special_tokens=True)[0]
    assistant_response = full_response.split(tokenizer.eos_token)[0].strip()

    print("\n\033[34mAssistant:\n \033[0m")
    print("\033[34m" + assistant_response + "\033[0m")

    nfe = STEPS  # 原生 LLaDA 固定步数
    num_tokens = len(tokenizer(assistant_response, add_special_tokens=False)["input_ids"])
    elapsed = end_time - start_time
    tps = num_tokens / elapsed if elapsed > 0 else 0
    tpf = num_tokens / nfe if nfe > 0 else 0
    print(f"\n[Stats] Tokens: {num_tokens} | Time: {elapsed:.2f}s | NFE: {nfe} | TPS: {tps:.2f} | TPF: {tpf:.2f}")
