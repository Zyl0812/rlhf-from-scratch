"""RLHF 第 1 阶段：Supervised Fine-Tuning（SFT）。

基于 `uer/gpt2-chinese-cluecorpussmall` 这个中文 GPT-2 在电商评论数据
`online_shopping_10_cats.csv` 上做下一词预测（causal LM）微调，得到
`gpt2-sft/`，作为后续 PPO 阶段的 actor 与 reference model 的初始化权重。

要点：
- 数据已经是"目标分布"的文本（用户评论），直接做 LM loss 即可，不需要
  prompt-response 拼接（InstructGPT 论文里的 SFT 是 instruction tuning，
  这里简化为纯领域语言建模，目的是让模型先学会说"电商口吻"）。
- 只训 1 个 epoch：再多容易过拟合，且后续 PPO 还要继续动这些参数，
  灾难性遗忘风险更高。
- 用 `DataCollatorForLanguageModeling(mlm=False)` 自动把 input_ids
  右移一位作为 labels，loss 仅在非 pad 位置计算。

运行：
    uv run python instructgpt/sft.py
依赖：仓库根目录下需放好 gpt2 预训练权重和评论数据集，路径见 BASE_DIR。
"""

from pathlib import Path
from pprint import pprint

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    pipeline,
    set_seed,
)

BASE_DIR = Path(__file__).resolve().parent
pretrained_model_path = BASE_DIR / "gpt2-chinese-cluecorpussmall"
dataset_path = BASE_DIR / "online_shopping_10_cats.csv"
output_dir = BASE_DIR / "gpt2-sft"

dataset = load_dataset("csv", data_files=str(dataset_path))

ds_train = dataset["train"]
# 将评论少于1024个字的过滤出来
ds_train = ds_train.filter(
    lambda x: x["review"] is not None and len(x["review"]) > 20 and len(x["review"]) < 1024
)

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
model = AutoModelForCausalLM.from_pretrained(pretrained_model_path)
max_length = model.config.n_positions


def tokenize(batch):
    return tokenizer(
        batch["review"],
        truncation=True,
        max_length=max_length,
    )


map_kwargs = {
    "batched": True,
    "batch_size": 512,
    "remove_columns": ["cat", "label", "review"],
}

tokenized_dataset_train = ds_train.map(tokenize, **map_kwargs)
# 转换成torch张量格式
tokenized_dataset_train.set_format(type="torch")
# 将pad_token作为eos_token
tokenizer.eos_token = tokenizer.pad_token
# 将数据整理成预测下一个token的任务的数据格式
data_collator = DataCollatorForLanguageModeling(
    tokenizer,
    mlm=False,  # 将数据整理成预测下一个token的格式
)

dataloader_params = {"batch_size": 2, "collate_fn": data_collator}

train_dataloader = DataLoader(tokenized_dataset_train, **dataloader_params)

# 要更新的是model的参数
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
# 一般sft会训练1个epoch，也就是把训练数据看一遍就可以了
# 否则容易过拟合，造成灾难性遗忘
num_epochs = 1

device = torch.device("cuda")
model.to(device)
for _epoch in range(num_epochs):
    model.train()
    for i, batch in enumerate(train_dataloader):
        batch = batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Step: {i}, Loss: {loss.item()}")

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)


# 测试微调后的模型
g = pipeline("text-generation", model=str(output_dir))
set_seed(42)
pprint(g("这本书真是", max_length=300, num_return_sequences=1))
