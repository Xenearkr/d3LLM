from datasets import load_dataset

# 1. 加载数据集
dataset = load_dataset("XenonGas/dream_coder_trajectory_ds1000")

# 2. 保存到本地目录
dataset.save_to_disk("/home/u-shengbf/Codes/d3LLM/trajectory_data_ds1000")

print("数据集已保存到本地。")