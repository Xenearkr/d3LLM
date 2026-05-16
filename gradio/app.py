"""
d3LLM Dream-Coder Gradio 可视化对话界面（流式对话 + 单步解码网格）。
启动: python gradio/app.py
"""
from __future__ import annotations

import html
import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

_GRADIO_DIR = Path(__file__).resolve().parent
_ROOT = _GRADIO_DIR.parent
for p in (str(_GRADIO_DIR), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from model_runner import (
    DEFAULT_MODEL,
    GenerationParams,
    get_runner,
    normalize_messages_for_chat_template,
)

# 解码网格：16 列，行数随 max_new_tokens 变化；单元格用 1fr 铺满容器
GRID_COLS = 16
GRID_GAP_PX = 4

FONT_UI = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", '
    'Arial, "Noto Sans", "Liberation Sans", sans-serif'
)
FONT_MONO = (
    'ui-monospace, "SF Mono", "Cascadia Code", "Segoe UI Mono", Menlo, Monaco, '
    'Consolas, "Liberation Mono", monospace'
)

PLACEHOLDER_VIZ = f"""
<div class="decode-viz-panel" style="width:100%;min-height:min(72vh,640px);padding:20px 16px;border-radius:12px;border:1px solid #e2e8f0;background:linear-gradient(180deg,#f8fafc 0%,#f1f5f9 100%);display:flex;align-items:center;justify-content:center;box-sizing:border-box;font-family:{FONT_UI};">
  <p style="margin:0;color:#64748b;font-size:14px;text-align:center;line-height:1.65;">发送消息后，此处<b>实时</b>展示当前 Step 的解码网格（16 列，行数随生成长度变化）。<br/>灰色为 mask；下方滑块可回看各步。</p>
</div>
"""

STATS_LOADING_HTML = f"""
<div class="stats-dashboard" style="font-family:{FONT_UI};">
  <div style="padding:14px 16px;border:1px solid #e0e7ff;border-radius:12px;background:linear-gradient(135deg,#eef2ff,#f8fafc);color:#4338ca;font-size:13px;font-weight:500;text-align:center;">正在加载模型…</div>
</div>
"""

STATS_PLACEHOLDER_HTML = f"""
<div class="stats-dashboard" style="font-family:{FONT_UI};">
  <div style="padding:14px 16px;border:1px solid #e2e8f0;border-radius:12px;background:#fff;box-shadow:0 1px 2px rgba(15,23,42,.04);color:#64748b;font-size:13px;text-align:center;">等待生成…</div>
</div>
"""


def _confidence_bg(conf: float) -> str:
    r = int(255 * (1 - conf))
    g = int(255 * conf)
    return f"rgb({r},{g},95)"


def _cell_label_text(tokenizer, tid: int, mask_id: int) -> Tuple[str, bool]:
    if tid == mask_id:
        return "▪", True
    s = tokenizer.decode([tid], skip_special_tokens=False)
    s = s.replace("\n", "↵").replace("\r", "").replace("\t", " ")
    s = s.strip() or "·"
    if len(s) > 8:
        s = s[:7] + "…"
    return s, False


def _stat_row(label: str, value: str, *, last: bool = False) -> str:
    border = "" if last else "border-bottom:1px solid #f1f5f9;"
    return (
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:10px;'
        f'padding:8px 0;{border}">'
        f'<span style="color:#64748b;font-size:12px;flex-shrink:0;">{html.escape(label)}</span>'
        f'<span style="font-weight:600;color:#0f172a;font-size:13px;text-align:right;'
        f'font-variant-numeric:tabular-nums;word-break:break-all;">{html.escape(value)}</span>'
        f"</div>"
    )


def _stat_section(title: str, rows: List[Tuple[str, str]], *, variant: str = "indigo") -> str:
    """variant: indigo | slate"""
    if variant == "indigo":
        head_bg = "linear-gradient(135deg,#eef2ff 0%,#e0e7ff 55%,#f8fafc 100%)"
        title_c = "#3730a3"
        bar = "#6366f1"
    else:
        head_bg = "linear-gradient(135deg,#f1f5f9 0%,#e2e8f0 50%,#f8fafc 100%)"
        title_c = "#334155"
        bar = "#64748b"
    inner = "".join(
        _stat_row(lab, val, last=(i == len(rows) - 1)) for i, (lab, val) in enumerate(rows)
    )
    return (
        f'<div style="border:1px solid #e2e8f0;border-radius:12px;overflow:hidden;'
        f'background:#fff;box-shadow:0 1px 3px rgba(15,23,42,.06);margin-bottom:10px;">'
        f'<div style="display:flex;align-items:center;gap:10px;padding:10px 14px;'
        f'background:{head_bg};border-bottom:1px solid #e2e8f0;">'
        f'<div style="width:3px;height:16px;border-radius:2px;background:{bar};flex-shrink:0;"></div>'
        f'<span style="font-weight:600;color:{title_c};font-size:13px;letter-spacing:.02em;">'
        f"{html.escape(title)}</span></div>"
        f'<div style="padding:4px 14px 10px;">{inner}</div></div>'
    )


def build_stats_dashboard_html(
    *,
    step_index: int,
    n_steps: int,
    nfe: int,
    masks_remaining: int,
    elapsed: float,
    done: bool,
    result=None,
) -> str:
    """与 Gradio Soft / indigo 主题一致的统计与参数面板（HTML）。"""
    if done and result is not None:
        p = result.params
        stat_rows = [
            ("生成 token 数", str(result.num_tokens)),
            ("耗时", f"{result.elapsed_sec:.2f}s"),
            ("NFE（前向次数）", str(result.nfe)),
            ("TPS", f"{result.tps:.2f}"),
            ("TPF", f"{result.tpf:.2f}"),
            ("轨迹步数", str(n_steps)),
        ]
        param_rows = [
            ("max_new_tokens", str(p["max_new_tokens"])),
            ("temperature", str(p["temperature"])),
            ("entropy threshold", str(p["threshold"])),
            ("block_size", str(p["block_size"])),
            ("block_add_threshold", str(p["block_add_threshold"])),
            ("decoded_token_threshold", str(p["decoded_token_threshold"])),
            ("early_stop", str(p["early_stop"])),
            ("alg", str(p["alg"])),
        ]
        return (
            f'<div class="stats-dashboard" style="font-family:{FONT_UI};">'
            + _stat_section("生成统计（已完成）", stat_rows, variant="indigo")
            + _stat_section("本次采样参数", param_rows, variant="slate")
            + "</div>"
        )

    prog_rows = [
        ("当前 Step", f"{step_index} / {max(n_steps - 1, 0)}"),
        ("NFE", str(nfe)),
        ("剩余 mask", str(masks_remaining)),
        ("已用时间", f"{elapsed:.2f}s"),
    ]
    return (
        f'<div class="stats-dashboard" style="font-family:{FONT_UI};">'
        + _stat_section("生成中…", prog_rows, variant="indigo")
        + "</div>"
    )


def build_error_stats_html(msg: str) -> str:
    esc = html.escape(msg)
    return f'''<div class="stats-dashboard" style="font-family:{FONT_UI};">
<div style="border:1px solid #fecaca;border-radius:12px;overflow:hidden;background:#fff;">
<div style="padding:10px 14px;background:linear-gradient(135deg,#fef2f2,#fff7ed);border-bottom:1px solid #fecaca;">
<span style="font-weight:600;color:#b91c1c;font-size:13px;">生成失败</span></div>
<div style="padding:12px 14px;color:#7f1d1d;font-size:12px;line-height:1.55;word-break:break-word;">{esc}</div>
<div style="padding:0 14px 12px;font-size:11px;color:#64748b;">多轮对话若仍失败，可点击「清空」后重试。已自动将消息中的非纯文本字段转为文本再送入模板。</div>
</div></div>'''


def build_single_step_grid_html(
    step: Optional[Dict[str, Any]],
    tokenizer,
    mask_token_id: int,
    grid_cols: int = GRID_COLS,
    total_slots: int = 256,
) -> str:
    """单 Step 解码网格：单元格随容器等分放大，贴近边缘。"""
    if step is None:
        return PLACEHOLDER_VIZ

    cols = max(8, min(int(grid_cols), 32))
    nrow = max(1, (total_slots + cols - 1) // cols)
    step_no = int(step.get("step", 0))
    n_new = int(step.get("num_decoded", 0))
    n_mask = int(step.get("masks_remaining", 0))
    ids = step.get("gen_token_ids") or []
    confs = step.get("gen_confidences") or []

    L = min(len(ids), total_slots)
    ids = list(ids[:total_slots])
    confs = list(confs[:total_slots]) + [None] * max(0, total_slots - len(confs))

    cells: List[str] = []
    for i in range(total_slots):
        if i < L:
            tid = int(ids[i])
            conf_opt = confs[i] if i < len(confs) else None
        else:
            tid = mask_token_id
            conf_opt = None

        text, is_mask = _cell_label_text(tokenizer, tid, mask_token_id)
        txt_esc = html.escape(text)
        if is_mask:
            bg, fg = "#cbd5e1", "#475569"
            title = f"#{i} · [mask]"
        else:
            conf = float(conf_opt) if conf_opt is not None else 0.85
            bg = _confidence_bg(conf)
            fg = "#0f172a"
            cstr = f"{conf:.3f}" if conf_opt is not None else "—"
            title = f"#{i} · id={tid} · conf={cstr}"

        cells.append(
            f'<div title="{html.escape(title)}" style="'
            "min-width:0;min-height:0;width:100%;height:100%;box-sizing:border-box;"
            "display:flex;align-items:center;justify-content:center;"
            f"background:{bg};color:{fg};"
            f"font-size:clamp(9px, min(1.9vw, 1.7vh), 13px);font-family:{FONT_MONO};"
            "border-radius:4px;border:1px solid rgba(15,23,42,.08);overflow:hidden;"
            'padding:2px 3px;line-height:1.15;text-align:center;">'
            f"{txt_esc}</div>"
        )

    cells_html = "".join(cells)
    gap = GRID_GAP_PX

    return (
        f'<div class="decode-viz-panel" style="width:100%;box-sizing:border-box;padding:12px 14px 14px;font-family:{FONT_UI};'
        'border-radius:12px;border:1px solid #e2e8f0;background:#fafafa;'
        'box-shadow:0 1px 3px rgba(15,23,42,.06);display:flex;flex-direction:column;">'
        '<div style="margin:0 6px 8px;display:flex;flex-wrap:wrap;align-items:center;gap:8px 12px;">'
        f'<span style="font-size:13px;font-weight:600;color:#1e293b;">Step {step_no}</span>'
        f'<span style="font-size:12px;color:#64748b;">本步新解 <b style="color:#4338ca;">{n_new}</b> · '
        f'剩余 mask <b style="color:#0f172a;">{n_mask}</b> · 共 {total_slots} 格</span></div>'
        '<div style="flex:1;min-height:min(72vh,820px);max-height:78vh;width:100%;overflow:auto;padding:12px 14px;box-sizing:border-box;">'
        f'<div style="display:grid;width:100%;height:100%;min-height:min(68vh,760px);'
        f"grid-template-columns:repeat({cols},minmax(0,1fr));"
        f"grid-template-rows:repeat({nrow},minmax(0,1fr));"
        f'gap:{gap}px;box-sizing:border-box;">'
        f"{cells_html}</div></div></div>"
    )


def _viz_session_from_state(state: Optional[dict]) -> dict:
    if not state:
        return {
            "trace_steps": [],
            "mask_id": 0,
            "grid_cols": GRID_COLS,
            "total_slots": 256,
            "model_path": DEFAULT_MODEL,
        }
    return state


def render_step_from_session(step_index: float, viz_state: dict) -> Tuple[str, int]:
    sess = _viz_session_from_state(viz_state)
    steps = sess.get("trace_steps") or []
    if not steps:
        return PLACEHOLDER_VIZ, 0
    idx = max(0, min(int(round(float(step_index))), len(steps) - 1))
    mp = sess.get("model_path") or DEFAULT_MODEL
    runner = get_runner(mp)
    if not runner._loaded:
        runner.load()
    tok = runner.tokenizer
    if tok is None:
        return PLACEHOLDER_VIZ, idx
    html_out = build_single_step_grid_html(
        steps[idx],
        tok,
        int(sess.get("mask_id", 0)),
        grid_cols=int(sess.get("grid_cols", GRID_COLS)),
        total_slots=int(sess.get("total_slots", 256)),
    )
    return html_out, idx


def load_model(model_path: str) -> str:
    return get_runner(model_path).load()


def _step_slider_update(*, logical_step: int, last_step_index: int):
    """Gradio Slider 要求 minimum < maximum；仅 Step 0 时 last_step_index==0，不能用 maximum=0。"""
    last = max(0, int(last_step_index))
    mx = max(last, 1)
    val = max(0, min(int(logical_step), last))
    return gr.update(minimum=0, maximum=mx, value=val)


def chat_respond_stream(
    message: str,
    history: List[Dict[str, str]],
    model_path: str,
    max_new_tokens: int,
    temperature: float,
    threshold: float,
    block_size: int,
    block_add_threshold: float,
    decoded_token_threshold: float,
    early_stop: bool,
    viz_state: dict,
    step_index: int,
):
    """流式：对话与解码网格同步更新；生成时自动跟随最新 Step。"""
    if not message.strip():
        yield (
            history,
            STATS_PLACEHOLDER_HTML,
            PLACEHOLDER_VIZ,
            viz_state,
            _step_slider_update(logical_step=0, last_step_index=0),
            0,
        )
        return

    runner = get_runner(model_path)
    if not runner._loaded:
        yield (
            history,
            STATS_LOADING_HTML,
            PLACEHOLDER_VIZ,
            viz_state,
            _step_slider_update(logical_step=0, last_step_index=0),
            0,
        )
        runner.load()

    norm_hist = normalize_messages_for_chat_template(history or [])
    messages = norm_hist + [{"role": "user", "content": str(message).strip()}]
    params = GenerationParams(
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        threshold=float(threshold),
        block_size=int(block_size),
        block_add_threshold=float(block_add_threshold),
        decoded_token_threshold=float(decoded_token_threshold),
        early_stop=bool(early_stop),
    )

    aligned_tokens = runner._align_max_new_tokens(params.max_new_tokens, params.block_size)
    mask_id = int(runner.model.config.mask_token_id)

    streaming_messages = messages + [{"role": "assistant", "content": "▌"}]
    viz_state = {
        "trace_steps": [],
        "mask_id": mask_id,
        "grid_cols": GRID_COLS,
        "total_slots": aligned_tokens,
        "model_path": model_path,
    }

    yield (
        streaming_messages,
        build_stats_dashboard_html(
            step_index=0, n_steps=1, nfe=0, masks_remaining=aligned_tokens, elapsed=0, done=False
        ),
        PLACEHOLDER_VIZ,
        viz_state,
        _step_slider_update(logical_step=0, last_step_index=0),
        0,
    )

    last_viz_html = PLACEHOLDER_VIZ
    last_max_step = 0
    try:
        for chunk in runner.generate_stream(messages, params):
            viz_state["trace_steps"] = chunk.trace_steps
            step = chunk.trace_steps[chunk.step_index]
            masks_rem = int(step.get("masks_remaining", 0))
            n_steps = len(chunk.trace_steps)
            max_step = max(0, n_steps - 1)

            streaming_messages[-1] = {
                "role": "assistant",
                "content": chunk.partial_text + ("▌" if not chunk.done else ""),
            }

            display_idx = max_step
            viz_html = build_single_step_grid_html(
                chunk.trace_steps[display_idx],
                runner.tokenizer,
                mask_id,
                grid_cols=GRID_COLS,
                total_slots=aligned_tokens,
            )
            last_viz_html = viz_html
            last_max_step = max_step

            stats = build_stats_dashboard_html(
                step_index=display_idx,
                n_steps=n_steps,
                nfe=chunk.nfe,
                masks_remaining=masks_rem,
                elapsed=chunk.elapsed_sec,
                done=chunk.done,
                result=chunk.result if chunk.done else None,
            )

            if chunk.done and chunk.result:
                streaming_messages[-1] = {
                    "role": "assistant",
                    "content": chunk.result.assistant_text,
                }

            yield (
                streaming_messages,
                stats,
                viz_html,
                viz_state,
                _step_slider_update(logical_step=display_idx, last_step_index=max_step),
                display_idx,
            )

            if chunk.done:
                break
    except Exception as e:
        streaming_messages[-1] = {
            "role": "assistant",
            "content": f"[生成出错] {str(e)}",
        }
        if viz_state.get("trace_steps"):
            try:
                idx = max(0, len(viz_state["trace_steps"]) - 1)
                last_viz_html = build_single_step_grid_html(
                    viz_state["trace_steps"][idx],
                    runner.tokenizer,
                    mask_id,
                    grid_cols=GRID_COLS,
                    total_slots=aligned_tokens,
                )
                last_max_step = max(0, len(viz_state["trace_steps"]) - 1)
            except Exception:
                pass
        yield (
            streaming_messages,
            build_error_stats_html(str(e)),
            last_viz_html,
            viz_state,
            _step_slider_update(logical_step=last_max_step, last_step_index=last_max_step),
            last_max_step,
        )


def on_step_slider_change(step_index: float, viz_state: dict):
    html_out, idx = render_step_from_session(step_index, viz_state)
    return html_out, idx


def clear_chat():
    empty_state = {
        "trace_steps": [],
        "mask_id": 0,
        "grid_cols": GRID_COLS,
        "total_slots": 256,
        "model_path": DEFAULT_MODEL,
    }
    return (
        [],
        STATS_PLACEHOLDER_HTML,
        PLACEHOLDER_VIZ,
        empty_state,
        _step_slider_update(logical_step=0, last_step_index=0),
        0,
        "",
    )


CUSTOM_CSS = f"""
.gradio-container, .gradio-container .wrap, .gradio-container .contain {{
    font-family: {FONT_UI} !important;
}}
.gradio-container label, .gradio-container input, .gradio-container textarea,
.gradio-container button, .gradio-container .markdown, .gradio-container .prose,
.gradio-container .tabitem, .gradio-container .block, .gradio-container span,
.gradio-container p, .gradio-container h1, .gradio-container h2, .gradio-container h3 {{
    font-family: {FONT_UI} !important;
    letter-spacing: 0.01em;
}}
.gradio-container .message-wrap, .gradio-container .message, .gradio-container .chatbot {{
    font-size: 15px !important;
    line-height: 1.65 !important;
}}
.main-title {{ text-align: center; margin-bottom: 0; font-weight: 600; }}
.sub-title {{ text-align: center; color: #64748b; font-size: 14px; margin-top: 0; line-height: 1.6; }}
.decode-viz-panel {{ width: 100%; font-family: {FONT_UI}; }}
.stats-dashboard {{ max-height: 70vh; overflow-y: auto; font-family: {FONT_UI}; }}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="d3LLM Dream-Coder 可视化") as demo:
        viz_state = gr.State(
            {
                "trace_steps": [],
                "mask_id": 0,
                "grid_cols": GRID_COLS,
                "total_slots": 256,
                "model_path": DEFAULT_MODEL,
            }
        )
        current_step = gr.State(0)

        gr.Markdown(
            "# diffusion LLM 解码可视化\n",
            elem_classes=["main-title"],
        )
        gr.Markdown(
            "左侧为流式对话；右侧为解码网格; 下方为生成参数与统计 / 采样参数。",
            elem_classes=["sub-title"],
        )

        with gr.Row():
            model_path = gr.Textbox(value=DEFAULT_MODEL, label="模型路径", scale=3)
            load_btn = gr.Button("加载模型", variant="primary", scale=1)
        load_status = gr.Textbox(label="模型状态", interactive=False)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=360):
                chatbot = gr.Chatbot(label="对话（流式）", height=520)
                msg = gr.Textbox(
                    label="输入消息",
                    placeholder="例如: Write a Python function to compute fibonacci.",
                    lines=2,
                )
                with gr.Row():
                    send_btn = gr.Button("发送", variant="primary")
                    clear_btn = gr.Button("清空")

                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        with gr.Accordion("生成参数", open=True):
                            max_new_tokens = gr.Slider(32, 512, value=256, step=32, label="max_new_tokens")
                            temperature = gr.Slider(0, 1.5, value=0.0, step=0.1, label="temperature")
                            threshold = gr.Slider(0.0, 2.0, value=0.45, step=0.05, label="entropy threshold")
                            block_size = gr.Slider(8, 64, value=32, step=8, label="block_size")
                            block_add_threshold = gr.Slider(
                                0.0, 1.0, value=1.0, step=0.1, label="block_add_threshold"
                            )
                            decoded_token_threshold = gr.Slider(
                                0.0, 1.0, value=1.0, step=0.1, label="decoded_token_threshold"
                            )
                            early_stop = gr.Checkbox(value=True, label="early_stop")
                    with gr.Column(scale=1):
                        stats_html = gr.HTML(value=STATS_PLACEHOLDER_HTML)

            with gr.Column(scale=1, min_width=420):
                viz_html = gr.HTML(value=PLACEHOLDER_VIZ)
                step_slider = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0,
                    step=1,
                    label="解码 Step（拖动查看各步状态；生成时自动跟随最新步）",
                )

        chat_inputs = [
            msg,
            chatbot,
            model_path,
            max_new_tokens,
            temperature,
            threshold,
            block_size,
            block_add_threshold,
            decoded_token_threshold,
            early_stop,
            viz_state,
            current_step,
        ]
        chat_outputs = [chatbot, stats_html, viz_html, viz_state, step_slider, current_step]

        load_btn.click(load_model, inputs=[model_path], outputs=[load_status])

        send_btn.click(
            chat_respond_stream,
            inputs=chat_inputs,
            outputs=chat_outputs,
        ).then(lambda: "", outputs=[msg])

        msg.submit(
            chat_respond_stream,
            inputs=chat_inputs,
            outputs=chat_outputs,
        ).then(lambda: "", outputs=[msg])

        step_slider.change(
            on_step_slider_change,
            inputs=[step_slider, viz_state],
            outputs=[viz_html, current_step],
        )

        clear_btn.click(
            clear_chat,
            outputs=[chatbot, stats_html, viz_html, viz_state, step_slider, current_step, msg],
        )

        demo.load(load_model, inputs=[model_path], outputs=[load_status])

    return demo


def _pick_server_port() -> int:
    preferred = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))

    def _bindable(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return True
            except OSError:
                return False

    if _bindable(preferred):
        return preferred
    for port in range(preferred + 1, preferred + 64):
        if _bindable(port):
            print(f"[Gradio] 端口 {preferred} 已被占用，改用 {port}")
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    app = build_app()
    _port = _pick_server_port()
    app.launch(
        server_name="0.0.0.0",
        server_port=_port,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="indigo"),
        css=CUSTOM_CSS,
    )
