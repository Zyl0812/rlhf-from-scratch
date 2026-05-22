"""RLHF 第 2 阶段：Reward Model（RM）训练。

InstructGPT 的标准做法是用 *人类成对偏好* 数据训 RM；本仓库为了简化复现，
改用现成的二分类情感标签（评论 → 0/1）作为"奖励信号"：
    正向评论 → 高分（1.0），负向评论 → 低分（0.0）
等价于把"用户喜不喜欢这段话"的人类偏好用情感标签近似。

RM 结构（`RewardModel`）：在 GPT-2 之上加一个 `nn.Linear(hidden, 1)`
"奖励头"，对最后一层 hidden state 的每个位置都打一个分。训练时只取
*最后一个 token*（这里专门追加了一个 eos 作为 "reward token"）的分数
做 BCE loss，避免逐 token 监督带来的噪声。

输出：`reward_model.pt` 是后续 PPO 阶段的奖励函数 R(x, y)。
"""

from pathlib import Path

import torch
from datasets import load_dataset
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding

BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "gpt2-chinese-cluecorpussmall"
dataset_path = BASE_DIR / "online_shopping_10_cats.csv"
reward_model_output_path = BASE_DIR / "reward_model.pt"
tokenizer = AutoTokenizer.from_pretrained(model_path)
max_length = AutoConfig.from_pretrained(model_path).n_positions - 1

tokenizer.eos_token = tokenizer.pad_token
REWARD_TOKEN_ID = tokenizer.eos_token_id

ds = load_dataset("csv", data_files=str(dataset_path))
ds_train = ds["train"]

ds_train = ds_train.filter(
    lambda x: x["review"] is not None and len(x["review"]) > 20 and len(x["review"]) < 1024
)

print("数据集的数量：", len(ds_train))


def tokenize(batch):
    # 提取出文本内容
    outputs = tokenizer(
        batch["review"],
        truncation=True,
        max_length=max_length,
    )
    outputs["score"] = [0] * len(outputs["input_ids"])
    # 对每条数据的最后的reward token进行评分
    outputs["score_index"] = [0] * len(outputs["input_ids"])
    for i in range(len(outputs["input_ids"])):
        # 第 i 条数据的末尾添加一个 eos token，作为reward token
        outputs["input_ids"][i].append(REWARD_TOKEN_ID)
        # reward token的掩码设置为 1 。
        outputs["attention_mask"][i].append(1)
        # 正向情感的文本评分为 1 。负向情感的评分为 0 。
        outputs["score"][i] = float(batch["label"][i])
        # 对 reward token 进行评分，也就是评分的索引为 reward token 的索引。
        outputs["score_index"][i] = len(outputs["input_ids"][i]) - 1
    return outputs


map_kwargs = {"batched": True, "batch_size": 512, "remove_columns": ["cat", "label", "review"]}

tokenized_dataset_train = ds_train.map(tokenize, **map_kwargs)

tokenized_dataset_train.set_format(type="torch")


class RewardModel(nn.Module):
    """GPT-2 主干 + 一个标量奖励头。

    forward 返回 (batch, seq_len) 形状的逐 token 奖励 logits；外部代码会
    根据 `score_index`（即每条样本的 reward token 位置）只取出一个标量
    作为整段文本的奖励。这种"只在最后一个 token 监督"的做法与 InstructGPT
    论文一致，理由是奖励应是对整段输出的整体评分，而非局部信号。
    """

    def __init__(self, model_name):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        # 奖励头：hidden_size → 1，对每个位置打一个标量分数
        self.reward_head = nn.Linear(self.llm.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # 必须打开 output_hidden_states 才能拿到最后一层 hidden state（默认只返回 logits）
        transformer_outputs = self.llm.forward(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        last_hidden_state = transformer_outputs.hidden_states[-1]  # (B, T, H)
        # (B, T, H) → reward_head → (B, T, 1) → squeeze → (B, T)
        reward = self.reward_head(last_hidden_state).squeeze(-1)
        return reward


model = RewardModel(model_path)

data_collator = DataCollatorWithPadding(tokenizer)

dataloader_params = {
    "batch_size": 2,  # 还是使用6G显存
    "shuffle": True,
    "collate_fn": data_collator,
}

train_dataloader = DataLoader(tokenized_dataset_train, **dataloader_params)

device = torch.device("cuda")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# 二分类交叉熵损失
criterion = nn.BCEWithLogitsLoss()
num_epochs = 1  # N+ Implementation Detail paper

model.to(device)

for _epoch in range(num_epochs):
    model.train()
    for i, batch in enumerate(train_dataloader):
        inputs = batch.to(device)
        model_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        # 模型针对训练数据的打分
        scores = model(**model_inputs)
        batch_indices = torch.arange(scores.shape[0], device=scores.device)
        # 模型对reward token的打分
        score = scores[batch_indices, inputs["score_index"]]
        # 真实分数：0或者1
        target = inputs["score"]
        loss = criterion(score, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("Step-", i, ", Loss: ", loss.item())

torch.save(model.state_dict(), reward_model_output_path)

model.eval()

all_predictions = []
all_labels = []

for _i, batch in enumerate(train_dataloader):
    inputs = batch.to(device)
    model_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
    with torch.no_grad():
        scores = model(**model_inputs)
        batch_indices = torch.arange(scores.shape[0], device=scores.device)
        score = scores[batch_indices, inputs["score_index"]]
        target = inputs["score"]
    prob = torch.sigmoid(score)
    predictions = (prob > 0.5).int()
    all_predictions.extend(predictions.cpu().numpy())
    all_labels.extend(target.cpu().numpy())

print(confusion_matrix(all_labels, all_predictions))
