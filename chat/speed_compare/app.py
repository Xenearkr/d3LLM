"""
Gradio 速度比较页面：并行比较 3 个模型的生成速度。

模型：
  1. XenonGas/finetune_d3LLM_DREAM_Coder   (DreamModel + 多块并行解码 generate_multi_block)
  2. Dream-org/Dream-Coder-v0-Instruct-7B  (AutoModel + 标准阈值扩散 diffusion_generate)
  3. Qwen/Qwen2.5-Coder-7B-Instruct        (自回归, TextIteratorStreamer 流式输出)

xenon 使用 DreamModel 加载并绑定 D3LLMGenerationMixin 方法，走多块流水线并行解码
（block_add_threshold / decoded_token_threshold 控制 block 推进节奏）。
vanilla 使用 AutoModel 加载，走标准逐步阈值扩散解码。
Qwen 走 transformers 的 TextIteratorStreamer 流式追加。
全部完成后在每个模型标题栏显示 Tokens / Time / TPS。

默认假设有多张 GPU：Dream 模型分别放到 cuda:0/1，Qwen 放到最后一张 GPU，
从而并行运行互不抢占、计时公平。GPU 数量不足时会自动回退到 cuda:0。

运行：
  conda activate diffusion-llm
  python chat/speed_compare/app.py
"""
import os

# 与其它 chat 脚本一致的 SSL 旁路（HF_DISABLE_SSL_VERIFY=1 时生效）
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
import time
import types
import threading
from pathlib import Path

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.utils_Dream.model.modeling_dream import DreamModel
from utils.utils_Dream.model.configuration_dream import DreamConfig
from d3llm.d3llm_DREAM.d3llm_dream_generate_util import DreamGenerationMixin as D3LLMGenerationMixin

import gradio as gr


# ---------------- 配置 ----------------

# (key, 显示名/仓库, 类型)
#   dream_mblock: DreamModel + generate_multi_block（多块并行解码）
#   dream:        AutoModel + diffusion_generate（标准阈值扩散）
#   qwen:         AutoModelForCausalLM + TextIteratorStreamer（自回归流式）
SPECS = [
    ("vanilla", "dream_coder",    "dream"),
    ("qwen",    "qwen2.5-coder",  "qwen"),
    ("xenon",   "d3LLM_Coder",    "dream_mblock"),
]

# key -> HuggingFace 仓库路径（加载模型用）
MODEL_REPO = {
    "vanilla": "Dream-org/Dream-Coder-v0-Instruct-7B",
    "qwen":    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "xenon":   "XenonGas/finetune_d3LLM_DREAM_Coder",
}

# 多块并行解码阈值（xenon）
MBLOCK_THRESHOLD = 0.5

# 启动时 warmup 的轮数（调试时可改小甚至 0 以加快重启）
WARMUP_ROUNDS = 0

# 标准扩散参数（vanilla Dream-Coder, 原始逐步熵扩散）
DREAM_PARAMS = dict(
    max_new_tokens=400,
    steps=256,
    temperature=0.0,
    top_p=None,
    alg="entropy",
    alg_temp=0.1,
    return_dict_in_generate=True,
)

# 多块并行解码参数（xenon, 与 chat_d3llm_dream_steps.py 一致）
MBLOCK_PARAMS = dict(
    attention_mask=None,
    max_new_tokens=400,
    output_history=False,
    return_dict_in_generate=True,
    steps=256,
    temperature=0.0,
    alg="entropy_threshold",
    threshold=1.5,
    block_size=32,
    block_add_threshold=0.2,
    decoded_token_threshold=0.5,
    cache_delay_iter=5,
    early_stop=True,
)

LOADED = {}  # key -> (kind, model, tokenizer, device) | None


# ---------------- 模型加载 ----------------

def load_all():
    ngpu = torch.cuda.device_count() or 1
    print(f"[info] 检测到 {ngpu} 张 GPU")
    for idx, (key, label, kind) in enumerate(SPECS):
        repo = MODEL_REPO[key]
        try:
            if kind == "dream_mblock":
                # 多块并行：用 DreamModel 加载 + 绑定 D3LLMGenerationMixin 方法
                dev = f"cuda:{idx % ngpu}"
                print(f"[load] {label} ({repo}) -> {dev}")
                config = DreamConfig.from_pretrained(repo, trust_remote_code=True)
                try:
                    config._attn_implementation = "flash_attention_2"
                except Exception:
                    pass
                m = DreamModel.from_pretrained(
                    repo, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16,
                )
                m = m.to(dev).eval()
                m.generate_multi_block = types.MethodType(D3LLMGenerationMixin.generate_multi_block, m)
                m._sample_multi_block = types.MethodType(D3LLMGenerationMixin._sample_multi_block, m)
                m._sample_multi_block_kv_cache = types.MethodType(D3LLMGenerationMixin._sample_multi_block_kv_cache, m)
                m._prepare_inputs = types.MethodType(D3LLMGenerationMixin._prepare_inputs, m)
                tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
                LOADED[key] = ("dream_mblock", m, tok, dev)
            elif kind == "dream":
                # 标准扩散：用 AutoModel 加载，调用 diffusion_generate
                dev = f"cuda:{idx % ngpu}"
                print(f"[load] {label} ({repo}) -> {dev}")
                m = AutoModel.from_pretrained(repo, torch_dtype=torch.bfloat16, trust_remote_code=True)
                tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
                m = m.to(dev).eval()
                LOADED[key] = ("dream", m, tok, dev)
            else:
                dev = f"cuda:{(ngpu - 1)}"  # 最后一张 GPU 给 Qwen
                print(f"[load] {label} ({repo}) -> {dev}")
                m = AutoModelForCausalLM.from_pretrained(
                    repo, torch_dtype=torch.bfloat16, trust_remote_code=True
                )
                tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
                m = m.to(dev).eval()
                LOADED[key] = ("qwen", m, tok, dev)
        except Exception as e:
            print(f"[load] {label} 加载失败: {e}")
            LOADED[key] = None

    # Warmup（与 chat_d3llm_dream.py / chat_d3llm_dream_coder.py 一致）：
    # 首次推理会触发 CUDA kernel 编译/图捕获，比稳态慢，不 warmup 会让首跑计时偏高。
    # 轮数由 WARMUP_ROUNDS 控制；运行中还可用界面 Warmup 按钮重新预热。
    warmup_questions = _load_warmup_questions()
    for key, label, kind in SPECS:
        entry = LOADED.get(key)
        if entry is None:
            continue
        try:
            n = min(WARMUP_ROUNDS, len(warmup_questions))
            if n <= 0:
                print(f"[warmup] {label} 跳过 (WARMUP_ROUNDS=0)")
                continue
            print(f"[warmup] {label} x{n}")
            for i in range(n):
                q = warmup_questions[i % len(warmup_questions)]
                _, m, tok, dev = entry
                if kind == "dream_mblock":
                    _warmup_mblock_round(m, tok, dev, q)
                elif kind == "dream":
                    _warmup_dream_round(m, tok, dev, q)
                else:
                    _warmup_qwen_round(m, tok, dev, q)
        except Exception as e:
            print(f"[warmup] {label} 预热失败: {e}")


def _warmup_mblock_round(m, tok, dev, q):
    """多块并行 Dream 模型跑一轮 warmup（阻塞）。"""
    inputs = tok(q, return_tensors="pt")
    input_ids = inputs["input_ids"].to(dev)
    with torch.no_grad():
        m.generate_multi_block(
            input_ids,
            generation_tokens_hook_func=lambda step, x, logits: x,
            **MBLOCK_PARAMS,
        )


def _warmup_dream_round(m, tok, dev, q):
    """标准扩散 Dream 模型跑一轮 warmup（阻塞）。"""
    inputs = tok(q, return_tensors="pt")
    input_ids = inputs["input_ids"].to(dev)
    attn = inputs.get("attention_mask", torch.ones_like(input_ids)).to(dev)
    with torch.no_grad():
        m.diffusion_generate(
            input_ids,
            attention_mask=attn,
            **{**DREAM_PARAMS, "generation_tokens_hook_func": lambda step, x, logits: x},
        )


def _warmup_qwen_round(m, tok, dev, q):
    """Qwen 跑一轮 warmup（阻塞，短生成）。"""
    messages = [{"role": "user", "content": q}]
    prompt_text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tok(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(dev)
    attn = inputs.get("attention_mask", torch.ones_like(input_ids)).to(dev)
    with torch.no_grad():
        m.generate(input_ids=input_ids, attention_mask=attn,
                   max_new_tokens=16, do_sample=False)


def _load_warmup_questions():
    path = Path(__file__).resolve().parents[2] / "utils" / "serve" / "test_question.txt"
    try:
        with open(path, "r") as f:
            content = f.read()
        qs = [q.strip() for q in content.split("\n\n") if q.strip()]
        return qs or ["Write a hello world program in Python."]
    except Exception as e:
        print(f"[warmup] 无法读取 {path}: {e}; 使用默认问题")
        return ["Write a hello world program in Python."] * 10


# ---------------- 生成（每个模型一个后台线程） ----------------

def _style_mask_tokens(text, mask_token="<mask>"):
    """将 mask token 替换为带灰色背景的 HTML span，使其在 Markdown 中可见。"""
    if not mask_token or mask_token not in text:
        return text
    return text.replace(mask_token, f'<span class="mask-token">{mask_token}</span>')


def _make_dream_hook(tokenizer, prompt_len, buf):
    mask_tok = getattr(tokenizer, "mask_token", None) or "<mask>"

    def hook(step, x, logits):
        # 扩散循环开始前会被以 (None, x, None) 调用一次，跳过
        if step is None:
            return x
        gen_ids = x[0][prompt_len:]
        # 不 skip 特殊 token，让 mask token 可见，直观体现逐步去噪
        txt = tokenizer.decode(gen_ids, skip_special_tokens=False)
        txt = txt.split(tokenizer.eos_token)[0]
        buf.append(_style_mask_tokens(txt, mask_tok))
        return x
    return hook


def start_dream(key, prompt, params):
    """标准扩散生成（diffusion_generate）。"""
    _, m, tok, dev = LOADED[key]
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tok(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(dev)
    attn = inputs.get("attention_mask", torch.ones_like(input_ids)).to(dev)
    prompt_len = input_ids.shape[1]

    buf = []
    state = {"done": False, "resp": "", "tokens": 0, "time": 0.0, "tps": 0.0, "error": None}
    hook = _make_dream_hook(tok, prompt_len, buf)

    def worker():
        try:
            t0 = time.time()
            with torch.no_grad():
                out = m.diffusion_generate(
                    input_ids,
                    attention_mask=attn,
                    generation_tokens_hook_func=hook,
                    **params,
                )
            elapsed = time.time() - t0
            full = tok.decode(out.sequences[0][input_ids.shape[1]:], skip_special_tokens=True)
            resp = full.split(tok.eos_token)[0].strip()
            num = len(tok.encode(resp, add_special_tokens=False))
            state.update(resp=resp, tokens=num, time=elapsed,
                         tps=num / elapsed if elapsed > 0 else 0.0)
        except Exception as e:
            state["error"] = str(e)
        finally:
            state["done"] = True

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return buf, state, t


def start_mblock(key, prompt, params):
    """多块并行解码生成（generate_multi_block）。"""
    _, m, tok, dev = LOADED[key]
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tok(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(dev)
    prompt_len = input_ids.shape[1]

    buf = []
    state = {"done": False, "resp": "", "tokens": 0, "time": 0.0, "tps": 0.0, "nfe": 0, "error": None}
    hook = _make_dream_hook(tok, prompt_len, buf)

    def worker():
        try:
            t0 = time.time()
            with torch.no_grad():
                out, nfe = m.generate_multi_block(
                    input_ids,
                    generation_tokens_hook_func=hook,
                    **params,
                )
            elapsed = time.time() - t0
            full = tok.decode(out.sequences[0], skip_special_tokens=True)
            resp = full.split(tok.eos_token)[0].split("assistant\n")[-1].strip()
            num = len(tok.encode(resp, add_special_tokens=False))
            state.update(resp=resp, tokens=num, time=elapsed, nfe=nfe,
                         tps=num / elapsed if elapsed > 0 else 0.0)
        except Exception as e:
            state["error"] = str(e)
        finally:
            state["done"] = True

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return buf, state, t


def start_qwen(key, prompt, max_new_tokens):
    _, model, tokenizer, dev = LOADED[key]
    buf = []
    state = {"done": False, "text": "", "tokens": 0, "time": 0.0, "tps": 0.0, "error": None}
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(dev)
    attn = inputs.get("attention_mask", torch.ones_like(input_ids)).to(dev)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    def worker():
        try:
            t0 = time.time()
            gen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                streamer=streamer,
            )
            # 在后台线程中调用 generate，streamer 会在生成过程中产出 token
            thread = threading.Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
            thread.start()

            acc = ""
            for new_text in streamer:
                acc += new_text
                buf.append(acc)  # 累积快照，与 Dream 的 buf 语义一致
            thread.join()

            elapsed = time.time() - t0
            ntok = len(tokenizer.encode(acc, add_special_tokens=False))
            state.update(text=acc, tokens=ntok, time=elapsed,
                         tps=ntok / elapsed if elapsed > 0 else 0.0)
        except Exception as e:
            state["error"] = str(e)
        finally:
            state["done"] = True

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return buf, state, t


# ---------------- 统计与对比表 ----------------

def format_stats(s):
    if s is None:
        return "—"
    return f"Tokens: {s['tokens']} | Time: {s['time']:.2f}s | TPS: {s['tps']:.2f}"


def build_table(records):
    valid = [(l, s) for l, s in records if s]
    if not valid:
        return "**无可用结果。**"
    max_tps = max(s["tps"] for _, s in valid)
    lines = ["| 模型 | Tokens | Time (s) | TPS |", "|---|---|---|---|"]
    for label, s in records:
        if s is None:
            lines.append(f"| {label} | — | — | — |")
        else:
            star = " 🏆" if (s["tps"] == max_tps and max_tps > 0) else ""
            lines.append(f"| {label}{star} | {s['tokens']} | {s['time']:.2f} | {s['tps']:.2f} |")
    return "\n".join(lines)


# ---------------- 主比较生成器（Gradio 流式） ----------------

def compare(prompt):
    int_max = DREAM_PARAMS["max_new_tokens"]
    int_steps = DREAM_PARAMS["steps"]
    dream_params = {**DREAM_PARAMS}
    mblock_params = {**MBLOCK_PARAMS}

    n = len(SPECS)
    streams = ["⏳ 等待开始..."] * n
    stats = [""] * n

    def snap():
        # 与 outputs 列表顺序一致：每个模型 [标题(含统计), 消息] 交错
        interleaved = []
        for i in range(n):
            header = f"**{SPECS[i][1]}**\n\n{stats[i]}"
            interleaved.append(header)
            interleaved.append(streams[i])
        return tuple(interleaved)

    # 并行启动所有已加载模型
    runners = []
    for idx, (key, label, kind) in enumerate(SPECS):
        if LOADED.get(key) is None:
            streams[idx] = "⚠️ 模型未加载（见终端日志）"
            stats[idx] = "—"
            runners.append(None)
            continue
        if kind == "dream_mblock":
            buf, state, t = start_mblock(key, prompt, mblock_params)
        elif kind == "dream":
            buf, state, t = start_dream(key, prompt, dream_params)
        else:
            buf, state, t = start_qwen(key, prompt, int_max)
        runners.append((buf, state, t))
        streams[idx] = "🔄 生成中..."

    yield snap()

    last = [0] * n
    while True:
        for idx, r in enumerate(runners):
            if r is None:
                continue
            buf, state, _ = r
            while last[idx] < len(buf):
                streams[idx] = buf[last[idx]]
                last[idx] += 1
        yield snap()
        if all((r is None or r[1]["done"]) for r in runners):
            break
        time.sleep(0.02)

    # 收尾：取最终回复 + 统计
    for idx, (key, label, kind) in enumerate(SPECS):
        r = runners[idx]
        if r is None:
            continue
        buf, state, t = r
        t.join()
        while last[idx] < len(buf):  # 排空残留
            streams[idx] = buf[last[idx]]
            last[idx] += 1
        if state.get("error"):
            streams[idx] = f"⚠️ 生成失败: {state['error']}"
            stats[idx] = "—"
        else:
            final_text = state.get("resp") or state.get("text") or streams[idx]
            # Dream 模型最终输出可能仍含 mask token，同样高亮
            _, _, kind = SPECS[idx]
            if kind.startswith("dream"):
                mask_tok = getattr(LOADED[key][2], "mask_token", None) or "<mask>" if LOADED.get(key) else "<mask>"
                final_text = _style_mask_tokens(final_text, mask_tok)
            streams[idx] = final_text
            s = {"tokens": state["tokens"], "time": state["time"], "tps": state["tps"]}
            stats[idx] = format_stats(s)

    yield snap()


# ---------------- Gradio UI ----------------

def build_ui():
    custom_css = """
    .model-header h3 {
        font-size: 1.4em !important;
        font-weight: 700 !important;
        margin: 4px 0 !important;
    }
    .model-content {
        font-size: 0.82em !important;
        line-height: 1.45 !important;
    }
    .model-content .mask-token {
        background-color: #d0d0d0;
        color: #555;
        border-radius: 3px;
        padding: 1px 3px;
        font-family: inherit;
    }
    """

    with gr.Blocks(title="Dream-Coder 速度比较", css=custom_css) as demo:
        prompt = gr.Textbox(
            label="Prompt",
            value="Write a hello world program in Python.",
            lines=2,
            scale=1,
        )

        outputs = []
        with gr.Row():
            for key, label, kind in SPECS:
                with gr.Column(scale=1, min_width=300):
                    header = gr.Markdown(f"### {label}", elem_classes=["model-header"])
                    st = gr.Markdown("", elem_classes=["model-content"])
                    outputs += [header, st]

        # 回车触发生成
        prompt.submit(compare, [prompt], outputs, concurrency_limit=1)

    return demo


if __name__ == "__main__":
    load_all()
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=False)
