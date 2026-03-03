# 观察不同 Diffusion Steps 的生成过程

在 HumanEval、MBPP 上使用 **Dream** 或 **LLaDA** 做代码生成时，可以通过以下两种方式观察「不同 diffusion steps」下的行为。

---

## 方式一：用本仓库脚本对比多种 steps（推荐）

使用 `observe_diffusion_steps.py` 在少量题目上，对**同一 prompt** 用多组 steps 生成，并对比输出与耗时；可选保存**逐步解码历史**（仅 Vanilla Dream 支持）。

### 1. 运行前

在 **d3LLM 仓库根目录**下执行（不要只在 `eval_scripts/` 下）：

```bash
cd /path/to/d3LLM   # 例如 ~/chenx/d3LLM/Codes/d3LLM
```

### 2. 基本用法

```bash
# Dream + HumanEval，默认对比 steps=8,32,64,128,256，并保存一次逐步历史
python eval_scripts/observe_diffusion_steps.py --model_type dream --dataset humaneval --save_history

# 只对比指定 steps，不保存逐步历史
python eval_scripts/observe_diffusion_steps.py --model_type dream --dataset mbpp --steps 8 32 64 256

# 使用 d3LLM-Dream（多块解码）
python eval_scripts/observe_diffusion_steps.py --model_type d3llm --dataset humaneval --steps 32 128 256

# 多跑几条题目、限制生成长度、指定输出目录
python eval_scripts/observe_diffusion_steps.py --model_type dream --dataset humaneval --num_samples 5 --max_new_tokens 128 --output_dir ./my_steps_output
```

### 3. 输出说明

- **compare_steps_{model_type}_{dataset}.json**  
  每个题目在不同 steps 下的：`response_preview`、`response_length`、`nfe`、`time_sec`。用于对比「步数 vs 质量/速度」。

- **step_history_*.txt**（仅当加 `--save_history` 且 `--model_type dream`）  
  某一道题的**逐步解码**：每步对应一次 “Step k” 下的当前生成文本，可直观看到从噪声到完整代码的演化。

---

## 方式二：用现有评估脚本扫不同 steps

不写新代码，只改 **diffusion_steps** 和 **output_path**，对 HumanEval/MBPP 做多组完整评估，再对比结果与 samples。

### Dream 系（lm_eval + humaneval_instruct / mbpp_instruct）

工作目录与环境（以你本机为准）：

```bash
export D3LLM_ROOT="$HOME/chenx/d3LLM/Codes/d3LLM"
cd "$D3LLM_ROOT/utils/utils_Dream/eval_instruct"
export PYTHONPATH="$D3LLM_ROOT/utils/utils_Dream/eval_instruct:$PYTHONPATH"
export HF_ALLOW_CODE_EVAL=1
```

HumanEval，例如 steps=256 / 128 / 64 / 32：

```bash
# steps=256
accelerate launch --main_process_port 12334 -m lm_eval --model diffllm \
  --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.1,top_p=0.9,alg=entropy \
  --tasks humaneval_instruct --device cuda --batch_size 1 --num_fewshot 0 \
  --output_path ./eval_tmp/humaneval_steps256 --log_samples --confirm_run_unsafe_code --apply_chat_template

# steps=128
accelerate launch --main_process_port 12335 -m lm_eval --model diffllm \
  --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=256,diffusion_steps=128,dtype=bfloat16,temperature=0.1,top_p=0.9,alg=entropy \
  --tasks humaneval_instruct --device cuda --batch_size 1 --num_fewshot 0 \
  --output_path ./eval_tmp/humaneval_steps128 --log_samples --confirm_run_unsafe_code --apply_chat_template
```

MBPP 同理，把 `--tasks humaneval_instruct` 改为 `--tasks mbpp_instruct`，并加上 `--num_fewshot 4`，再为不同 steps 指定不同 `--output_path`（如 `./eval_tmp/mbpp_steps256` 等）。

对比方式：看各 `output_path` 下的 **results_*.json**（pass@1 等）和 **samples_*.jsonl**（同一题在不同 steps 下的生成内容）。

### LLaDA 系（eval_llada.py）

LLaDA 的步数由 `model_args` 里的 **steps** / **gen_length** 控制。在 `eval_scripts/llada_humaneval.sh` 或 `llada_mbpp.sh` 里找到对应命令，复制多份，只改 `steps=256` → `steps=128` / `64` / `32` 以及 `output_path`，即可得到多组「不同 diffusion steps」的评估结果与样本，用于观察生成过程与指标变化。

---

## 小结

| 目标 | 做法 |
|------|------|
| 快速看「同一题、多组 steps」的生成与耗时 | 用 **observe_diffusion_steps.py**（方式一） |
| 看某一次生成的**逐步解码过程**（噪声→完整序列） | 方式一 + `--save_history`（仅 Vanilla Dream） |
| 做**整表评估**（pass@1 等）并对比不同 steps | 用现有 **lm_eval / eval_llada** 脚本扫多组 steps（方式二） |
