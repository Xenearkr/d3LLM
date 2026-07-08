# d3LLM Dream-Coder 部署与接入指南

本文档记录将 `XenonGas/finetune_d3LLM_DREAM_Coder` 经 **SGLang** 部署为 OpenAI 兼容 API，并从 **Windows 客户端**（Cursor / Codex++）接入的完整工作流。供在新服务器（如 A6000：`210.28.134.201`）上复现。

---

## 1. 架构概览

```
Windows (Cursor / Codex++)
    │  HTTPS
    ▼
cloudflared Quick Tunnel (Windows 本机，推荐)
    │  http://localhost:13000
    ▼
SSH 隧道 (-L 13000:127.0.0.1:30000)
    ▼
A6000 服务器 SGLang (:30000)
    ▼
d3LLM Dream-Coder 7B (扩散式代码模型)
```

**关键结论（踩坑总结）：**

| 场景 | 结论 |
|------|------|
| Cursor BYOK | 必须 **公网 HTTPS**；`127.0.0.1` / 内网 IP 不可用 |
| SSH 隧道单独给 Cursor | ❌ 无效（Cursor 云端转发，访问不了本机） |
| 服务器 ngrok 免费版 + 防火墙 | ❌ 直连超时；走代理需付费（ERR_NGROK_9009） |
| 服务器 cloudflared | ❌ 防火墙下 cloudflared 不走代理，Quick Tunnel 失败 |
| **Windows cloudflared + SSH** | ✅ **推荐方案** |
| ngrok 免费版 | 需 header `ngrok-skip-browser-warning: 1`；Codex/Cursor 不便 |
| Cursor Agent / Ask / Composer | ❌ 不走 BYOK 自定义 Base URL |
| Cursor Chat (`Ctrl+L`) | ✅ 可用 BYOK |
| Codex++ 全项目上下文 | 易超 8192 token 或 OOM；测试时用新会话 + 短 prompt |

---

## 2. 服务器环境准备（A6000）

### 2.1 硬件与系统

- GPU：NVIDIA A6000（48GB 显存，比 24GB 卡更适合 d3LLM）
- 系统：Linux + CUDA
- 仓库路径示例：`/home/<user>/Codes/d3LLM`

### 2.2 Conda 环境

使用已有 `diffusion-llm` 环境（或等价环境），需包含：

- Python 3.10+
- `sglang`（含 dLLM / Dream 支持）
- `torch` + CUDA

```bash
# 若环境已存在，激活即可
conda activate diffusion-llm

# 验证
python -c "import sglang; print('sglang ok')"
```

SGLang 安装参考（EvalPlus 脚本注释中的 PR 版本）：

```bash
pip install uv
uv pip install "sglang[all] @ git+https://github.com/sgl-project/sglang.git@refs/pull/20615/head#subdirectory=python"
```

### 2.3 模型权重

模型 HuggingFace ID：`XenonGas/finetune_d3LLM_DREAM_Coder`（约 15GB，4 分片）

**方式 A：脚本下载并软链**

```bash
bash eval_scripts/setup_d3llm_coder_model.sh
```

**方式 B：手动**

```bash
huggingface-cli download XenonGas/finetune_d3LLM_DREAM_Coder \
  --local-dir models/finetune_d3LLM_DREAM_Coder
```

**方式 C：从 HF 缓存软链**（若已下载过）

```bash
ln -sfn ~/.cache/huggingface/hub/models--XenonGas--finetune_d3LLM_DREAM_Coder/snapshots/<snapshot_id> \
  models/finetune_d3LLM_DREAM_Coder
```

验证权重完整：

```bash
ls models/finetune_d3LLM_DREAM_Coder/model-*.safetensors | wc -l   # 应为 4
```

> **注意**：不要用 `transformers 5.x` 直接 `AutoModel.from_pretrained` 测 Dream 模型（RoPE 兼容问题）。以 SGLang 启动为准。

---

## 3. 启动 SGLang 服务

### 3.1 脚本

```bash
cd /path/to/d3LLM

# 默认：GPU0，端口 30000，context 8192
CUDA_VISIBLE_DEVICES=0 \
ATTENTION_BACKEND=triton \
DISABLE_CUDA_GRAPH=1 \
bash eval_scripts/serve_d3llm_coder_sglang.sh
```

### 3.2 重要环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `MODEL_PATH` | `models/finetune_d3LLM_DREAM_Coder` | 模型目录 |
| `PORT` | `30000` | API 端口 |
| `API_KEY` | `sk-d3llm-local` | OpenAI 兼容 API Key |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU 编号 |
| `ATTENTION_BACKEND` | `triton` | 避免 flashinfer `-lcuda` 编译问题 |
| `DISABLE_CUDA_GRAPH` | `1` | 同上 |
| `CONTEXT_LENGTH` | `8192` | 上下文上限；A6000 可试 `16384` |
| `MEM_FRACTION_STATIC` | `0.80` | 显存占用比例 |
| `PYTHON` | 自动检测 | Python 解释器路径 |
| `BIND_HOST` | `0.0.0.0` | 勿用 `$HOST`（conda 会污染） |

A6000 上若需更长上下文（Codex 仍建议控制输入长度）：

```bash
CONTEXT_LENGTH=16384 MEM_FRACTION_STATIC=0.75 \
CUDA_VISIBLE_DEVICES=0 ATTENTION_BACKEND=triton DISABLE_CUDA_GRAPH=1 \
bash eval_scripts/serve_d3llm_coder_sglang.sh
```

### 3.3 冒烟测试

```bash
curl -s http://127.0.0.1:30000/v1/models \
  -H "Authorization: Bearer sk-d3llm-local"

curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H "Authorization: Bearer sk-d3llm-local" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "d3llm-dream-coder",
    "messages": [{"role":"user","content":"Write a Python hello world function"}],
    "max_tokens": 256,
    "stream": false
  }'
```

期望：返回 JSON，`choices[0].message.content` 含 Python 代码。

### 3.4 已知问题

- **短 prompt**（如 `hi`）可能返回空 `content`；用具体编程题测试。
- **超长输入**（>8192 token）会报错或被 OOM 杀死；Codex 需新会话、不带全仓库。
- **OOM 后服务挂掉**：需重新运行 `serve_d3llm_coder_sglang.sh`。

---

## 4. 公网暴露（Windows 侧，推荐）

因多数实验室服务器无法直连 ngrok/Cloudflare edge，**在 Windows 本机跑隧道**。

### 4.1 前置：SSH 隧道

PowerShell（保持窗口打开）：

```powershell
ssh -N -L 13000:127.0.0.1:30000 <user>@210.28.134.201
```

验证：

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:13000/v1/models" `
  -Headers @{ Authorization = "Bearer sk-d3llm-local" }
```

### 4.2 Cloudflare Quick Tunnel（推荐）

Windows 安装 cloudflared：

```powershell
winget install Cloudflare.cloudflared
# 或下载 exe 后用完整路径运行
```

启动（新 PowerShell 窗口，保持运行）：

```powershell
cloudflared tunnel --url http://localhost:13000
```

记录输出 URL，例如：`https://xxxx.trycloudflare.com`

验证：

```powershell
Invoke-RestMethod -Uri "https://xxxx.trycloudflare.com/v1/models" `
  -Headers @{ Authorization = "Bearer sk-d3llm-local" }
```

> Quick Tunnel URL 重启后会变，需更新客户端 Base URL。

### 4.3 ngrok（备选，不推荐 Codex/Cursor）

```powershell
ngrok http 13000
```

免费版会返回 HTML 警告页（ERR_NGROK_6024），API 客户端需加 header：

```
ngrok-skip-browser-warning: 1
```

Cursor/Codex 通常无法加此 header，故 **优先 Cloudflare**。

### 4.4 服务器侧 cloudflared（仅当服务器能直连 Cloudflare 时）

```bash
# 安装
curl -fsSL -o ~/bin/cloudflared \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x ~/bin/cloudflared

bash eval_scripts/serve_d3llm_coder_cloudflare.sh
```

防火墙环境通常失败；有公网直连的 A6000 可尝试。

---

## 5. 客户端配置

### 5.1 API 参数（统一）

| 项 | 值 |
|----|-----|
| Base URL | `https://<tunnel-host>/v1` |
| API Key | `sk-d3llm-local` |
| Model | `d3llm-dream-coder` |

### 5.2 Cursor

1. `Ctrl+Shift+J` → **Models**
2. 开启 **Override OpenAI Base URL** → 填入 `https://....trycloudflare.com/v1`
3. **OpenAI API Key** → `sk-d3llm-local`
4. **`Ctrl+L` Chat**（不要用 Agent / Composer / Ask）
5. 模型名输入：`d3llm-dream-coder`

### 5.3 Codex++

- Base URL：`https://....trycloudflare.com/v1`
- API Key：`sk-d3llm-local`
- Model：`d3llm-dream-coder`
- **新建会话**，发具体编程题；避免附带整个项目（否则 >8192 token 或 OOM）

---

## 6. EvalPlus 评测（可选）

先启动 SGLang，再：

```bash
export SGLANG_SERVED_MODEL=d3llm-dream-coder
bash eval_scripts/evalplus_by_sglang.sh humaneval
bash eval_scripts/evalplus_by_sglang.sh mbpp
```

结果目录：`evalplus_results_sglang/`

---

## 7. 运维命令速查

```bash
# 查看 SGLang 是否在跑
curl -s http://127.0.0.1:30000/v1/models -H "Authorization: Bearer sk-d3llm-local"

# 查看日志
tail -f models/sglang_d3llm_coder.log

# 停止 SGLang
pkill -f "sglang.launch_server.*finetune_d3LLM_DREAM_Coder"

# GPU 显存
nvidia-smi
```

---

## 8. 迁移到 A6000 检查清单

- [ ] 克隆仓库 `git clone ... d3LLM`
- [ ] 创建/激活 `diffusion-llm` conda 环境并安装 sglang
- [ ] 下载模型 `bash eval_scripts/setup_d3llm_coder_model.sh`
- [ ] 启动 SGLang（A6000 可试更大 `CONTEXT_LENGTH`）
- [ ] 服务器本地 curl 冒烟通过
- [ ] Windows SSH 隧道 `13000→30000`
- [ ] Windows cloudflared → 获得公网 URL
- [ ] Cursor Chat / Codex 配置 Base URL + Key + 模型名
- [ ] 用编程题（非 hi）测试

---

## 9. 文件清单（本次部署相关）

| 文件 | 用途 |
|------|------|
| `eval_scripts/serve_d3llm_coder_sglang.sh` | 启动 SGLang API |
| `eval_scripts/serve_d3llm_coder_cloudflare.sh` | 服务器侧 Cloudflare 隧道（可选） |
| `eval_scripts/setup_d3llm_coder_model.sh` | 下载/软链模型 |
| `eval_scripts/evalplus_by_sglang.sh` | EvalPlus 经 SGLang 评测 |
| `models/README.md` | 模型目录说明 |
| `docs/deploy_d3llm_dream_coder.md` | 本文档 |
