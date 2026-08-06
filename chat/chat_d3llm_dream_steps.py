"""
d3LLM-Dream 对话脚本 —— 流式逐步扩散可视化版本（基于 chat_d3llm_dream.py）。

与 chat_d3llm_dream.py 的唯一区别：利用 generate_multi_block 的
generation_tokens_hook_func 回调（已加到 _sample_multi_block，默认 no-op，不影响
原有行为），每一次 forward（nfe +1）后立即清空上一步的屏幕输出并打印当前步的
中间结果（带步骤编号 [step nfe]），全部步骤跑完后再输出最终回复与统计信息。
其余逻辑（模型加载、multi_block 生成参数、warmup、最终解码、stats）保持不变。

若从 Hugging Face 下载模型时出现 Connection reset by peer，可使用国内镜像：
  export HF_ENDPOINT=https://hf-mirror.com
  或运行: HF_ENDPOINT=https://hf-mirror.com python chat/chat_d3llm_dream_steps.py
"""
import sys
from pathlib import Path
import math
import shutil
import types
import time
import torch
import tqdm
from transformers import AutoTokenizer

# Add project root to sys.path to enable imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.utils_Dream.model.modeling_dream import DreamModel
from utils.utils_Dream.model.configuration_dream import DreamConfig

# Add d3llm_DREAM for multi-block generation (already added above)
from d3llm.d3llm_DREAM.d3llm_dream_generate_util import DreamGenerationMixin as D3LLMGenerationMixin

# Model path
m = "d3LLM/d3LLM_Dream"
# m = "d3LLM/d3LLM_Dream_Coder"

tokenizer = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
device = torch.device("cuda:0")

# Load Dream model using original DreamModel (not dInfer wrapper)
print("Loading Dream model...")
model_config = DreamConfig.from_pretrained(m, trust_remote_code=True)

# Enable Flash Attention 2 in config
try:
    model_config._attn_implementation = "flash_attention_2"
    print("Flash Attention 2 configuration set")
except Exception as e:
    print(f"Warning: Could not set Flash Attention 2 in config: {e}")

model = DreamModel.from_pretrained(
    m,
    config=model_config,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)
model = model.to(device).eval()

# 先添加所有自定义方法（必须在编译前添加）
model.generate_multi_block = types.MethodType(D3LLMGenerationMixin.generate_multi_block, model)
model._sample_multi_block = types.MethodType(D3LLMGenerationMixin._sample_multi_block, model)
model._sample_multi_block_kv_cache = types.MethodType(D3LLMGenerationMixin._sample_multi_block_kv_cache, model)
model._prepare_inputs = types.MethodType(D3LLMGenerationMixin._prepare_inputs, model)

# torch.compile 会包装模型，导致动态添加的方法无法直接访问
# 解决方案：不编译模型，或通过 _orig_mod 访问（但更复杂）
# 为简化，默认禁用编译；如需启用，需确保方法通过 _orig_mod 访问
import os
if os.getenv("TORCH_COMPILE_DISABLE", "1") != "0":  # 默认禁用
    print("torch.compile disabled (default). To enable: export TORCH_COMPILE_DISABLE=0")
else:
    try:
        print("Compiling model with torch.compile...")
        print("  Note: After compilation, custom methods may need to be accessed via model._orig_mod")
        # 使用更保守的编译模式
        model = torch.compile(model, mode="default", fullgraph=False)
        # 编译后，需要通过 _orig_mod 访问原始模型的方法
        # 但为了简化，我们保持原模型引用用于方法调用
        print("Model compilation complete.")
    except Exception as e:
        print(f"Warning: torch.compile failed ({e}), continuing without compilation.")
        print("  To disable compilation permanently, set: export TORCH_COMPILE_DISABLE=1")

# Multi-block generation parameters
multi_block_params = {
    "attention_mask": None,
    "max_new_tokens": 256,
    "output_history": False,
    "return_dict_in_generate": True,
    "steps": 256,
    "temperature": 0.,
    "alg": "entropy_threshold",
    "threshold": 0.5,
    "block_size": 32,
    "block_add_threshold": 0.1,
    "decoded_token_threshold": 0.95,
    "cache_delay_iter": 10000,
    "early_stop": True,
}

print("\n" + "="*80)
print("d3LLM-Dream Chat Mode (streaming steps, type 'quit' or 'exit' to end)")
print("="*80)

# Warmup model
test_questions_path = Path(__file__).parent.parent / 'utils' / 'serve' / 'test_question.txt'
test_questions = []
try:
    with open(test_questions_path, 'r') as f:
        content = f.read()
        # Split by empty lines to get individual questions
        questions = [q.strip() for q in content.split('\n\n') if q.strip()]
        test_questions = questions
except Exception as e:
    print(f"Warning: Could not load test questions: {e}. Using fallback warmup.")
    test_questions = ["Write a hello world program in Python."] * 10

with torch.no_grad():
    num_warmups = min(10, len(test_questions))
    for i in tqdm.tqdm(range(num_warmups), desc="Warming up model"):
        prompt_text = test_questions[i % len(test_questions)]
        inputs = tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs.get('attention_mask', torch.ones_like(input_ids)).to(device)
        output, nfe = model.generate_multi_block(
            input_ids,
            **multi_block_params
        )
print("Warmup complete.\n")

print("\033[31mNote that because our distillation data primarily consists of **coding** and **math reasoning** tasks, acceleration may only appear on prompts of these tasks.\033[0m")


def _visual_lines(text, cols):
    """估算 text 在终端中占用的视觉行数（考虑自动换行）。"""
    total = 0
    for ln in text.split("\n"):
        n = len(ln)
        total += max(1, math.ceil(n / cols)) if n else 1
    return total


def make_stream_hook(prompt_len):
    """构造每次 forward 后的流式打印回调。"""
    state = {"prev_lines": 0}

    def hook(step, x, logits):
        cols = shutil.get_terminal_size().columns or 80
        # 仅取生成部分；不 skip 特殊 token，让 mask token 可见，直观体现逐步去噪
        gen_ids = x[0][prompt_len:]
        step_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        step_text = step_text.split(tokenizer.eos_token)[0]
        display = f"[step {step}] {step_text}"
        n_lines = _visual_lines(display, cols)

        # 清空上一步打印的内容：光标上移 prev_lines 行，再清到屏幕末尾
        if state["prev_lines"] > 0:
            sys.stdout.write(f"\033[{state['prev_lines']}A")
            sys.stdout.write("\033[J")
        sys.stdout.write(display + "\n")
        sys.stdout.flush()
        state["prev_lines"] = n_lines
        return x

    def clear_last():
        """全部步骤结束后，清掉最后一步的屏幕输出。"""
        if state["prev_lines"] > 0:
            sys.stdout.write(f"\033[{state['prev_lines']}A")
            sys.stdout.write("\033[J")
            sys.stdout.flush()
            state["prev_lines"] = 0

    return hook, clear_last


messages = []
while True:
    # Get user input
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    if not user_input:
        continue

    # Add user message to conversation history
    messages = [
        {"role": "user", "content": user_input}
    ]

    # Prepare input
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt_text, return_tensors="pt")['input_ids'].to(device)
    prompt_len = input_ids.shape[1]

    stream_hook, clear_last = make_stream_hook(prompt_len)

    # Generate response
    print()  # 流式输出起始空行
    start_time = time.time()
    with torch.no_grad():
        output, nfe = model.generate_multi_block(
            input_ids,
            generation_tokens_hook_func=stream_hook,  # 每次 forward 后流式打印
            **multi_block_params,
        )
    end_time = time.time()

    # 清掉最后一步的中间输出，改为打印最终回复 + 统计信息
    clear_last()

    # Decode response
    full_response = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
    # Extract only the assistant's last response
    assistant_response = full_response.split(tokenizer.eos_token)[0].split("assistant\n")[-1].strip()

    print("\n\033[34mAssistant:\n \033[0m")
    print("\033[34m" + assistant_response + "\033[0m")

    # Calculate statistics - use tokenizer to count actual tokens
    num_generated_tokens = len(tokenizer.encode(assistant_response, add_special_tokens=False))
    elapsed_time = end_time - start_time
    tps = num_generated_tokens / elapsed_time if elapsed_time > 0 else 0  # Token per second
    tpf = num_generated_tokens / nfe if nfe > 0 else 0  # Token per forward

    print(f"\n[Stats] Tokens: {num_generated_tokens} | Time: {elapsed_time:.2f}s | "
          f"NFE: {nfe} | TPS (token/s): {tps:.2f} | TPF (token/forward): {tpf:.2f}")
