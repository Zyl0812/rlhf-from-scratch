# GRPO 路线：Qwen2.5 + Countdown（DeepSeek-R1 风格）

按 [DeepSeekMath (2024)](https://arxiv.org/abs/2402.03300) 提出的 GRPO 算法 + [DeepSeek-R1 (2025)](https://arxiv.org/abs/2501.12948) 的"规则奖励"思路，在 Countdown 算术任务上对 Qwen2.5-3B-Instruct 做强化学习对齐。

## 文件

| 文件 | 内容 |
|---|---|
| `train.py` | 主入口：DataLoader + 训练循环 + TensorBoard + 每 10 步 evaluate |
| `grpo.py` | 核心算法：`rollout` / `normalize_rewards_per_group` / `update_policy` |
| `qwen2_model.py` | **手写 Qwen2** Transformer：RoPE、RMSNorm、GQA、KV Cache、SafeTensors 加载、checkpoint_sequential |
| `countdown_task.py` | 任务定义、prompt 模板、`format_reward` + `answer_reward` |
| `tokenizer.py` | jinja2 chat template 封装（不依赖 transformers） |
| `data_types.py` | `Episode` / `MiniBatch` 数据结构 |

## 运行

```bash
# 准备：把 Qwen2.5-3B-Instruct/ 和 Countdown-Tasks-3to4/ 放在本目录下
uv run python grpo/train.py

# 看训练曲线
uv run tensorboard --logdir grpo/logs/
```

## 关键超参（写在 `train.py:main()`）

| 参数 | 值 | 含义 |
|---|---|---|
| `BATCH_SIZE` | 256 | 单 step 的 trajectory 数 |
| `NUM_QUESTIONS_PER_BATCH` | 32 | 每 batch 几道题 |
| `NUM_ANSWERS_PER_QUESTION` | 8 | 每道题采几条回答（即 GRPO 的"组大小"） |
| `max_gen_len` | 1024 | 训练时生成长度 / 评估时 2048 |
| `micro_batch_size` | 2 | 策略更新的梯度累积粒度 |
| `lr` | 1e-5 | AdamW |
| `max_grad_norm` | 1.0 | 梯度裁剪 |
| `dtype` | bf16 | 训练 + 推理统一 |

## GRPO vs PPO 一句话

> 同一道题采 N 条回答，组内 Z-score 当优势，省掉 critic。

```
PPO:   A_t = GAE_λ(r_t - V(s_t))              ← V 是另一个网络
GRPO:  A_i = (R_i - mean(R_group)) / std(R_group)
```

## 奖励设计（`countdown_task.py`）

| 项 | 取值 |
|---|---|
| `format_reward` 完全匹配 `<think>...</think>\n<answer>...</answer>` | 1.0 |
| 只有 `<think>` | 0.1 |
| 只有 `<answer>` | 0.5 |
| `answer_reward`：表达式合法 + 每个数字各用一次 + 求值等于 target | 1.0 |
| 任一条件不满足 | 0.0 |
| **总奖励** | `0.1 × format_reward + answer_reward` |

答案权重远大于格式权重，避免模型只学套壳。
