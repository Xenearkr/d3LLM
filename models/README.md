# 模型权重目录

本目录用于存放推理权重，**不纳入 git**（体积过大）。

## d3LLM Dream-Coder

```bash
bash eval_scripts/setup_d3llm_coder_model.sh
```

或手动指定路径后启动 SGLang：

```bash
export MODEL_PATH=/path/to/finetune_d3LLM_DREAM_Coder
bash eval_scripts/serve_d3llm_coder_sglang.sh
```

## 运行时产生的日志

以下文件由脚本自动生成，已在 `.gitignore` 中忽略：

- `sglang_d3llm_coder.log`
- `cloudflare_d3llm_coder.*`

完整部署说明见 [docs/deploy_d3llm_dream_coder.md](../docs/deploy_d3llm_dream_coder.md)。
