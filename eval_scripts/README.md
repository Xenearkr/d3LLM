# Evaluation Scripts

### Supported Methods

We include comprehensive evaluation code for:

- ✅ **d3LLM** (our method)
- ✅ [**AR Model (e.g., Qwen-2.5-7B-it)**](https://arxiv.org/abs/2412.15115) - Autoregressive baselines
- ✅ [**Vanilla LLaDA**](https://arxiv.org/abs/2502.09992) - Original LLaDA model
- ✅ [**Vanilla Dream**](https://arxiv.org/abs/2508.15487) - Original Dream model
- ✅ [**Fast-dLLM**](https://arxiv.org/abs/2505.22618) - Training-free acceleration with KV cache
- ✅ [**D2F**](https://arxiv.org/abs/2508.09192) - Discrete diffusion forcing
- ✅ [**dParallel**](https://arxiv.org/abs/2509.26488) - Distilled dLLMs
- ✅ [**Fast-dLLM v2**](https://arxiv.org/abs/2509.26328) - Block-wise diffusion

### Supported Benchmarks

```bash
# GSM8K
bash dream_gsm8k_cot.sh
bash llada_gsm8k_cot.sh

# MATH
bash dream_math.sh
bash llada_math.sh

# Code Generation (HumanEval & MBPP)
bash dream_humaneval.sh
bash dream_mbpp.sh
bash llada_humaneval.sh
bash llada_mbpp.sh
bash dream-coder.sh

# Long-Context GSM8K
bash dream_long_gsm8k.sh
bash llada_long_gsm8k.sh
```

---

### 在 HumanEval / MBPP 上评估现有 dLLM

支持在 **HumanEval** 与 **MBPP** 上评估以下四类 dLLM：**d3LLM-Dream**、**d3LLM-LLaDA**、**Vanilla Dream**、**Vanilla LLaDA**。

**方式一：使用便捷脚本（推荐）**

在仓库根目录或 `eval_scripts/` 下执行：

```bash
# 用法: bash eval_scripts/run_code_eval.sh <模型> <数据集>
# 模型: d3llm_dream | d3llm_llada | vanilla_dream | vanilla_llada
# 数据集: humaneval | mbpp

bash eval_scripts/run_code_eval.sh d3llm_dream humaneval
bash eval_scripts/run_code_eval.sh d3llm_dream mbpp
bash eval_scripts/run_code_eval.sh d3llm_llada humaneval
bash eval_scripts/run_code_eval.sh d3llm_llada mbpp
bash eval_scripts/run_code_eval.sh vanilla_dream humaneval
bash eval_scripts/run_code_eval.sh vanilla_dream mbpp
bash eval_scripts/run_code_eval.sh vanilla_llada humaneval
bash eval_scripts/run_code_eval.sh vanilla_llada mbpp
```

需先设置 `HF_ALLOW_CODE_EVAL=1`（脚本内会提示）；HumanEval 会执行生成的代码进行 pass@1 评估，MBPP 同理。

**方式二：从现有脚本中摘取单条命令**

- **Dream 系**（HumanEval / MBPP）  
  见 `dream_humaneval.sh`、`dream_mbpp.sh`：每个注释块对应一种模型配置（Vanilla Dream、d3LLM-Dream、Fast-dLLM、dParallel 等），复制对应块中的 `accelerate launch -m lm_eval ...` 在 `utils/utils_Dream/eval_instruct` 下执行即可。
- **LLaDA 系**（HumanEval / MBPP）  
  见 `llada_humaneval.sh`、`llada_mbpp.sh`：每个块对应一种模型（Vanilla LLaDA、d3LLM-LLaDA 等），在 `utils/utils_LLaDA` 下执行对应 `accelerate launch ... eval_llada.py ...` 即可。

**结果位置**

- Dream：`utils/utils_Dream/eval_instruct/eval_tmp/` 或 `--output_path` 指定目录下的 `samples_*`。
- LLaDA：`utils/utils_LLaDA/evals_results/<output_dir>/` 下会生成 `samples_*.jsonl`，HumanEval 会再跑 `postprocess_code_humaneval.py`，MBPP 跑 `postprocess_code_mbpp.py` 得到最终指标。
