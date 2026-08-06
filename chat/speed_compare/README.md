# Dream-Coder 速度比较 (Gradio)

并行比较 4 个模型的生成速度，并在界面上实时展示生成过程。

## 模型

| 面板 | 模型 | 框架 | 展示方式 |
|---|---|---|---|
| 1 | `XenonGas/finetune_d3LLM_DREAM_Coder` | Dream-Coder (diffusion) | 逐步扩散，每步覆盖上一步 |
| 2 | `d3LLM/d3LLM_Dream_Coder` | Dream-Coder (diffusion) | 逐步扩散，每步覆盖上一步 |
| 3 | `Dream-org/Dream-Coder-v0-Instruct-7B` | Dream-Coder (diffusion) | 逐步扩散，每步覆盖上一步 |
| 4 | `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF` | 自回归 (llama-cpp) | 流式追加 |

前三个模型走 `diffusion_generate` + `generation_tokens_hook_func`，每一步扩散结束后把当前解码结果（含 mask token，体现逐步去噪）刷新到面板；Qwen 走 `llama_cpp` 的 `create_chat_completion(stream=True)` 流式追加。全部完成后输出每个模型的 `Tokens / Time / TPS`，并汇总成对比表（最快 TPS 标 🏆）。

## 运行

```bash
conda activate diff
# Qwen GGUF 需要（前三个 Dream 模型不需要）：
pip install llama-cpp-python

# 可选：HF 镜像 / SSL 旁路
export HF_ENDPOINT=https://hf-mirror.com
export HF_DISABLE_SSL_VERIFY=1

python chat/speed_compare/app.py
# 浏览器打开 http://<host>:7860
```

## GPU 分配

默认按 GPU 数量自动分配，互不抢占、计时公平：

- 检测到 N 张 GPU 时，前三个 Dream 模型分别放到 `cuda:0 / cuda:1 / cuda:2`（不足则回退 `cuda:0`）。
- Qwen GGUF 放到最后一张 GPU（`tensor_split` 只给该 GPU 分层）。

当前机器为 4× RTX 6000 Ada (48GB)，四个模型各占一张 GPU 并行运行。若只有 1 张 GPU，四个模型会挤在 `cuda:0` 上并行，计时不再纯粹（但仍可对比）。

## 可调参数

- `app.py` 顶部 `QWEN_FILENAME`：Qwen GGUF 的量化文件名，可改为 `qwen2.5-coder-7b-instruct-q5_k_m.gguf` / `q8_0` 等。
- `DREAM_PARAMS`：Dream-Coder 框架的生成参数（与 `chat_d3llm_dream_coder_steps.py` 一致）。
- 界面上的 `Max new tokens` / `Diffusion steps` 滑块。

## 依赖

- gradio (已在 `diff` 环境)
- torch / transformers (已在 `diff` 环境)
- llama-cpp-python (需自行安装，仅 Qwen 面板需要)

## 备注

- 前三个 Dream 模型依赖各自 HF 仓库自带的 `generation_utils.py` 提供 `generation_tokens_hook_func` 钩子（与 vanilla Dream 一致）。若某仓库的远程代码未含该钩子，对应面板会显示 `⚠️ 生成失败`。
- Qwen 的 token 数用 `llm.tokenize` 统计（GGUF 自带 tokenizer），与 Dream 模型用各自 tokenizer 统计口径一致——均为「该模型自己生成的 token 数」，适合横向比 TPS。
