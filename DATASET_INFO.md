# d3LLM DREAM 训练数据集说明

## 📊 使用的数据集

`d3llm_dream_train.py` 训练脚本会使用 **两个数据集**：

### 1. 主训练数据集（问答对数据）

**数据集名称**: `Zigeng/dParallel_Dream_Distill_Data`

**来源**: HuggingFace Hub

**下载位置**: 
- 在线自动下载：训练脚本会自动从 HuggingFace 下载
- HuggingFace 链接: https://huggingface.co/datasets/Zigeng/dParallel_Dream_Distill_Data

**数据内容**:
- 包含 `question` 和 `llm_response` 字段
- 用于训练模型的问答对数据
- 主要用于代码和数学推理任务

**代码位置**:
```python
# 在 d3llm_dream_train.py 第 723 行
dataset = load_dataset("Zigeng/dParallel_Dream_Distill_Data", split="train")
```

### 2. 轨迹数据集（伪轨迹数据）

**数据集名称**: `d3LLM/trajectory_data_dream_32`

**来源**: HuggingFace Hub

**下载位置**:
- 在线自动下载：训练脚本会自动从 HuggingFace 下载
- HuggingFace 链接: https://huggingface.co/datasets/d3LLM/trajectory_data_dream_32
- 配置文件路径: `d3llm/d3llm_DREAM/distill_2_training/d3llm_train.yaml` 中的 `distillation.trajectory_dataset_path`

**数据内容**:
- 包含教师模型的解码顺序（伪轨迹）
- 每个样本包含多个轨迹步骤（从完全掩码到干净文本）
- 用于指导学生模型学习高效的生成模式
- 只使用 `is_correct=True` 的样本

**代码位置**:
```python
# 在 d3llm_dream_train.py 第 664 行
trajectory_dataset = load_dataset(trajectory_dataset_path, split="train")
# 第 669 行：过滤只保留正确的样本
trajectory_dataset = trajectory_dataset.filter(lambda x: x["is_correct"], num_proc=num_proc)
```

## 📥 数据集下载方式

### 方式 1: 自动下载（推荐）

训练脚本会自动从 HuggingFace 下载数据集，无需手动操作：

```bash
# 设置 HuggingFace 镜像（可选，加速下载）
export HF_ENDPOINT="https://hf-mirror.com"

# 运行训练脚本，会自动下载数据集
bash run_d3llm_dream_train.sh
```

### 方式 2: 手动预下载

如果需要提前下载数据集：

```bash
conda activate d3llm
python -c "
from datasets import load_dataset

# 下载主数据集
print('下载主数据集...')
main_dataset = load_dataset('Zigeng/dParallel_Dream_Distill_Data', split='train')
print(f'主数据集大小: {len(main_dataset)}')

# 下载轨迹数据集
print('下载轨迹数据集...')
trajectory_dataset = load_dataset('d3LLM/trajectory_data_dream_32', split='train')
print(f'轨迹数据集大小: {len(trajectory_dataset)}')
"
```

### 方式 3: 使用本地路径

如果已经下载到本地，可以修改配置文件使用本地路径：

```yaml
# d3llm_train.yaml
distillation:
    trajectory_dataset_path: "/path/to/local/trajectory_data_dream_32"  # 使用绝对路径
```

## 💾 数据集缓存位置

### HuggingFace 缓存

默认缓存位置（Linux）:
```
~/.cache/huggingface/datasets/
```

### 项目缓存

训练脚本会在以下位置创建缓存：

1. **预处理后的轨迹数据集缓存**:
   ```
   {trajectory_dataset_path}/cache/trajectory_preprocessed_{hash}.pkl
   ```

2. **分词后的数据集缓存**:
   ```
   d3llm/d3llm_DREAM/cache/tokenized_dataset_{hash}.pkl
   ```

## 🔍 数据集结构

### 主数据集结构

```python
{
    "question": str,        # 问题文本
    "llm_response": str,    # LLM 回答文本
    # ... 其他字段
}
```

### 轨迹数据集结构

```python
{
    "trajectory": List[List[int]],  # 轨迹步骤列表，每个步骤是一个 token ID 序列
    "is_correct": bool,             # 是否为正确答案
    # ... 其他字段
}
```

## ⚙️ 数据集配置

在 `d3llm_train.yaml` 中的相关配置：

```yaml
distillation:
    trajectory_dataset_path: "d3LLM/trajectory_data_dream_32"  # 轨迹数据集路径
    max_length: 512                    # 最大序列长度
    max_samples: null                  # 最大样本数（null = 使用全部，可设置如 1000 用于测试）
    num_proc: 8                        # 数据处理进程数
```

## 📝 注意事项

1. **网络连接**: 首次运行需要稳定的网络连接下载数据集
2. **磁盘空间**: 确保有足够的磁盘空间存储数据集和缓存
3. **镜像加速**: 如果下载慢，可以使用 HuggingFace 镜像：
   ```bash
   export HF_ENDPOINT="https://hf-mirror.com"
   ```
4. **测试模式**: 可以使用 `max_samples=1000` 进行快速测试，避免下载完整数据集
5. **缓存机制**: 数据集下载后会缓存，后续运行不需要重新下载

## 🔗 相关链接

- 主数据集: https://huggingface.co/datasets/Zigeng/dParallel_Dream_Distill_Data
- 轨迹数据集: https://huggingface.co/datasets/d3LLM/trajectory_data_dream_32
- LLaDA 轨迹数据集: https://huggingface.co/datasets/d3LLM/trajectory_data_llada_32
- 项目 README: https://github.com/hao-ai-lab/d3LLM

## 🐛 常见问题

### Q: 数据集下载失败怎么办？

A: 
1. 检查网络连接
2. 使用镜像: `export HF_ENDPOINT="https://hf-mirror.com"`
3. 手动下载后使用本地路径

### Q: 如何查看数据集大小？

A: 
```bash
conda activate d3llm
python -c "from datasets import load_dataset; ds = load_dataset('Zigeng/dParallel_Dream_Distill_Data'); print(f'主数据集: {len(ds[\"train\"])} 样本')"
```

### Q: 可以使用自己的数据集吗？

A: 可以，但需要：
1. 数据集格式与上述结构一致
2. 修改代码中的数据加载部分
3. 确保轨迹数据集格式正确
