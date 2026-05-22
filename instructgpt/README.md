# InstructGPT 路线：SFT → RM → PPO

按 [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) §3.3 的三阶段流程，对中文 GPT-2 做完整的 RLHF 对齐复现。

## 文件

| 文件 | 阶段 | 说明 |
|---|---|---|
| `sft.py` | 1. SFT | 在电商评论上做 next-token 微调，得到 `gpt2-sft/` |
| `reward_model.py` | 2. RM | GPT-2 + 奖励头，用情感二分类近似偏好奖励，得到 `reward_model.pt` |
| `ppo.py` | 3. PPO | Actor-Critic + KL 惩罚 + GAE + clip loss，用 `gpt2-sft` 做 actor 初始化，`reward_model.pt` 做奖励信号 |

## 运行（顺序依赖）

```bash
# 准备：把 gpt2-chinese-cluecorpussmall/ 和 online_shopping_10_cats.csv 放在本目录下
uv run python instructgpt/sft.py
uv run python instructgpt/reward_model.py
uv run python instructgpt/ppo.py
```

## 与论文实现的简化

| 论文 | 本仓库 |
|---|---|
| 人类标注的成对偏好数据训 RM | 用情感二分类标签（正/负向 → 1/0）近似 |
| 多 prompt 数据集 + 复杂 prompt 模板 | 截取评论的前 2~8 个 token 作为 prompt |
| 完整 PPO 训练循环 + 学习率调度 | 仅跑 100 个 step 用于验证流程 |

简化是为了在单卡（6G 显存即可）上跑通整条管线、便于学习。**算法骨架（KL 惩罚 / GAE / clip loss / mini-batch）与论文一致**。

## 核心算法在 `ppo.py` 中的位置

| 概念 | 函数 / 行号 |
|---|---|
| Actor-Critic 模型 | `class ActorCriticModel` |
| 冻结 ref model（计算 KL 用） | `ref_model = deepcopy(model)` |
| 逐 token 奖励 = -β·KL | `compute_rewards()` |
| GAE 优势估计 | `compute_advantage()` |
| Masked whitening | `masked_whiten()` |
| PPO clip loss + value loss | `compute_loss()` |
| Mini-batch 多轮更新 | `ppo_update()` |
