"""
基于 d3LLM Dream-Coder 的多轮对话脚本。
默认使用 Hugging Face 模型：d3LLM/d3LLM_Dream_Coder。
如需使用原生 Dream-Coder，可将 MODEL_PATH 改为 Dream-org/Dream-Coder-v0-Instruct-7B。
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

# 允许从项目根目录导入工具代码（如有需要）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Dream-Coder 模型（默认使用 d3LLM 版本）
MODEL_PATH = "d3LLM/d3LLM_Dream_Coder"
# 如需原生 Dream-Coder，请改为：
# MODEL_PATH = "Dream-org/Dream-Coder-v0-Instruct-7B"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Loading Dream-Coder model from {MODEL_PATH} ...")
model = AutoModel.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = model.to(device).eval()

# 生成参数（与 demo_multiturn_chat 类似，适当缩短以便交互）
MAX_NEW_TOKENS = 256
STEPS = 256

print("\n" + "=" * 80)
print(f"Dream-Coder Chat (model = {MODEL_PATH}). Type 'quit' or 'exit' to end.")
print("=" * 80)

# 预热（Warmup）
test_questions_path = Path(__file__).parent.parent / "utils" / "serve" / "test_question.txt"
test_questions = []
try:
    with open(test_questions_path, "r") as f:
        content = f.read()
        test_questions = [q.strip() for q in content.split("\n\n") if q.strip()]
except Exception as e:
    print(f"Warning: Could not load test questions: {e}. Using fallback warmup.")
    test_questions = ["Write a simple Python function to add two numbers."] * 10

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
            temperature=0.0,
            top_p=None,
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
            temperature=0.0,
            top_p=None,
            alg="entropy",
            alg_temp=0.1,
            return_dict_in_generate=True,
        )
    end_time = time.time()

    # 解码回复
    full_response = tokenizer.decode(output.sequences[0][input_ids.shape[1] :], skip_special_tokens=True)
    # 简单截取 assistant 内容
    assistant_response = full_response.split(tokenizer.eos_token)[0].strip()

    print("\n\033[34mAssistant:\n \033[0m")
    print("\033[34m" + assistant_response + "\033[0m")

    # 简单统计
    num_tokens = len(tokenizer.encode(assistant_response, add_special_tokens=False))
    elapsed = end_time - start_time
    tps = num_tokens / elapsed if elapsed > 0 else 0.0
    print(f"\n[Stats] Tokens: {num_tokens} | Time: {elapsed:.2f}s | TPS: {tps:.2f}")

