# d3LLM DREAM 训练指南

## 快速开始

### 1. 环境准备

确保已经创建并配置了 conda 环境：

```bash
cd ~/Codes/d3LLM
bash setup_env.sh
```

或者手动激活环境：

```bash
conda activate d3llm
```

### 2. 运行训练

使用提供的脚本运行训练：

```bash
cd ~/Codes/d3LLM
bash run_d3llm_dream_train.sh
```

或者直接使用 deepspeed 命令：

```bash
cd ~/Codes/d3LLM
conda activate d3llm
deepspeed --num_gpus=3 d3llm/d3llm_DREAM/distill_2_training/d3llm_dream_train.py
```

### 3. 自定义配置

可以通过命令行参数覆盖配置文件中的设置：

```bash
deepspeed --num_gpus=3 d3llm/d3llm_DREAM/distill_2_training/d3llm_dream_train.py \
    training.learning_rate=0.00001 \
    training.per_device_train_batch_size=2 \
    distillation.max_samples=1000
```

## 配置文件说明

配置文件位置：`d3llm/d3llm_DREAM/distill_2_training/d3llm_train.yaml`

主要配置项：

- **model.name**: 模型名称（默认: "Dream-org/Dream-v0-Instruct-7B"）
- **training.output_dir**: 输出目录
- **training.num_train_epochs**: 训练轮数（默认: 3）
- **training.per_device_train_batch_size**: 每设备批次大小（默认: 4）
- **training.gradient_accumulation_steps**: 梯度累积步数（默认: 4）
- **distillation.trajectory_dataset_path**: 轨迹数据集路径
- **distillation.max_samples**: 最大样本数（null = 使用全部数据）
- **lora.enabled**: 是否启用 LoRA（默认: true）

## 注意事项

1. **GPU 内存**: 如果遇到 OOM 错误，可以：
   - 减小 `per_device_train_batch_size`
   - 增加 `gradient_accumulation_steps`
   - 减小 `distillation.max_length`

2. **数据集**: 训练脚本会自动从 HuggingFace 下载数据集：
   - 主数据集: `Zigeng/dParallel_Dream_Distill_Data`
   - 轨迹数据集: `d3LLM/trajectory_data_dream_32`

3. **输出目录**: 训练输出会保存在 `output_model/d3LLM_DREAM_{SLURM_JOB_ID}_{TIMESTAMP}/`

4. **W&B 日志**: 如果使用 W&B，确保已登录：
   ```bash
   wandb login
   ```

5. **测试模式**: 可以设置 `distillation.max_samples=1000` 进行快速测试

## 故障排除

### 问题 1: CUDA out of memory
- 解决方案: 减小批次大小或启用梯度检查点

### 问题 2: 数据集下载失败
- 解决方案: 设置 `export HF_ENDPOINT="https://hf-mirror.com"` 使用镜像

### 问题 3: DeepSpeed 初始化失败
- 解决方案: 检查 CUDA 和 PyTorch 版本是否兼容

### 问题 4: 找不到轨迹数据集
- 解决方案: 检查 `distillation.trajectory_dataset_path` 路径是否正确

## 参考

- 项目 README: `README.md`
- 训练脚本: `d3llm/d3llm_DREAM/distill_2_training/d3llm_dream_train.py`
- 配置文件: `d3llm/d3llm_DREAM/distill_2_training/d3llm_train.yaml`
