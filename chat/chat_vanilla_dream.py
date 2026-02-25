"""
原生 Dream 对话脚本（与 chat_d3llm_dream.py 相同交互方式）。
使用模型：Dream-org/Dream-v0-Instruct-7B（与 Dream 项目一致）。
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 原生 Dream 模型（与 Dream 项目一致）
MODEL_PATH = "Dream-org/Dream-v0-Instruct-7B"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Loading vanilla Dream model...")
model = AutoModel.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = model.to(device).eval()

# 生成参数（原生 Dream 无 early_stop，步数固定）
MAX_NEW_TOKENS = 128
STEPS = 128

print("\n" + "=" * 80)
print("Vanilla Dream Chat (Dream-org/Dream-v0-Instruct-7B). Type 'quit' or 'exit' to end.")
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
        inputs = tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)
        _ = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            steps=STEPS,
            temperature=0.4,
            top_p=0.95,
            alg="entropy",
            alg_temp=0.1,
            return_dict_in_generate=True,
        )
print("Warmup complete.\n")

messages = []
while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break
    if not user_input:
        continue

    messages = [{"role": "user", "content": user_input}]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)

    start_time = time.time()
    with torch.no_grad():
        output = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            steps=STEPS,
            temperature=0.4,
            top_p=0.95,
            alg="entropy",
            alg_temp=0.1,
            return_dict_in_generate=True,
        )
    end_time = time.time()

    full_response = tokenizer.decode(output.sequences[0][input_ids.shape[1] :], skip_special_tokens=True)
    assistant_response = full_response.split(tokenizer.eos_token)[0].strip()
    if "assistant" in full_response:
        assistant_response = full_response.split("assistant")[-1].split(tokenizer.eos_token)[0].strip()

    print("\n\033[34mAssistant:\n \033[0m")
    print("\033[34m" + assistant_response + "\033[0m")

    nfe = STEPS  # 原生 Dream 固定步数，无 early_stop
    num_tokens = len(tokenizer.encode(assistant_response, add_special_tokens=False))
    elapsed = end_time - start_time
    tps = num_tokens / elapsed if elapsed > 0 else 0
    tpf = num_tokens / nfe if nfe > 0 else 0
    print(f"\n[Stats] Tokens: {num_tokens} | Time: {elapsed:.2f}s | NFE: {nfe} | TPS: {tps:.2f} | TPF: {tpf:.2f}")
