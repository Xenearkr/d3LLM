import os
# 设置镜像加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset

# 1. 加载远程数据集 (它会自动处理 .arrow 并解析为 Dataset 对象)
print("正在从镜像站加载并缓存数据...")
dataset = load_dataset("XenonGas/dream_coder_trajectory_apps")

# 2. 保存为包含 state.json 的完整格式
target_path = "/home/u-shengbf/Codes/d3LLM/trajectory_data_apps"

print(f"正在转换格式并保存到: {target_path}")
# 注意：如果目录已存在且非空，save_to_disk 可能会报错，建议先清空该目录
dataset.save_to_disk(target_path)

print("保存成功！你现在可以查看该目录下的文件了。")