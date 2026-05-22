# rlhf-from-scratch

> 从零复现 RLHF 全流程：GPT-2 SFT + Reward Model + PPO，以及 Qwen2.5 + Countdown 任务的 GRPO 训练（DeepSeek-R1 风格）

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📖 项目背景

本仓库是 **RLHF 与 GRPO 两条主流对齐路线** 的从零复现，覆盖从原始数据到训练曲线的完整链路。围绕两篇代表性论文做对照实现：

| 路线 | 论文 | 复现内容 |
|---|---|---|
| **InstructGPT 路线** | [Training language models to follow instructions with human feedback (OpenAI, 2022)](https://arxiv.org/abs/2203.02155) | 中文 GPT-2 的 SFT → RM → PPO 三阶段 |
| **DeepSeek-R1 路线** | [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs (2025)](https://arxiv.org/abs/2501.12948)、[DeepSeekMath（GRPO 提出）](https://arxiv.org/abs/2402.03300) | Qwen2.5-3B 在 Countdown 算术任务上的规则奖励 + GRPO |

附带 `model_notes/` 里 GPT-2 与 Qwen3 的**手写底层实现**（RoPE / GQA / RMSNorm / KV Cache），便于在面试或讲解中拆解模型内部细节。

适合读者：想完整看懂 RLHF/GRPO 训练管线、希望逐行追踪算法实现细节的同学。

---

## 🗂 仓库结构

```
rlhf-from-scratch/
├── README.md
├── pyproject.toml             # 依赖 + ruff 配置
├── .gitignore
├── instructgpt/               # InstructGPT 三阶段（PyTorch + transformers）
│   ├── README.md
│   ├── sft.py                 # 阶段 1：监督微调
│   ├── reward_model.py        # 阶段 2：奖励模型（情感二分类近似偏好）
│   └── ppo.py                 # 阶段 3：Actor-Critic + KL 惩罚 + GAE + PPO clip
├── grpo/                      # GRPO（手写 Qwen2 Transformer）
│   ├── README.md
│   ├── train.py               # 训练主循环 + TensorBoard 日志 + 评估
│   ├── grpo.py                # rollout / 组内标准化 / 微批次策略更新
│   ├── qwen2_model.py         # 手写 Qwen2：RoPE / GQA / RMSNorm / KV Cache / SafeTensors
│   ├── countdown_task.py      # 任务定义 + 格式奖励 + 答案正确性奖励
│   ├── tokenizer.py           # jinja2 chat template 封装
│   └── data_types.py          # Episode / MiniBatch
└── model_notes/               # 模型底层实现笔记（不参与 RLHF 训练）
    ├── README.md
    ├── gpt_2.py               # 手写 GPT-2（MHA + 可学位置编码 + LayerNorm + 朴素 KV Cache）
    └── qwen_3.py              # 手写 Qwen3（GQA + QK-Norm + RoPE + RMSNorm）
```

---

## 🚀 快速开始

### 环境准备

需要 Python 3.12，依赖管理用 [uv](https://docs.astral.sh/uv/)：

```bash
# 1. 安装 uv（已装可跳过）
# 见 https://docs.astral.sh/uv/getting-started/installation/

# 2. 同步依赖（自动创建 .venv）
uv sync

# 3. 验证环境
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 数据 / 模型下载

仓库**不打包**大文件，按需自行准备（路径写在各脚本顶部，可直接修改）：

| 资源 | 用途 | 获取方式 |
|---|---|---|
| `gpt2-chinese-cluecorpussmall/` | InstructGPT 三阶段的基座 | [HuggingFace: uer/gpt2-chinese-cluecorpussmall](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall) |
| `online_shopping_10_cats.csv` | SFT + RM 训练数据 | 公开电商评论情感数据集，搜索 "online_shopping_10_cats" |
| `Qwen2.5-3B-Instruct/` | GRPO 基座 | [HuggingFace: Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) |
| `Countdown-Tasks-3to4/` | GRPO 训练数据（parquet） | [HuggingFace: Jiayi-Pan/Countdown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4) |

---

## 🧪 InstructGPT 路线（三阶段）

### 阶段 1：SFT

```bash
uv run python instructgpt/sft.py
```

- 基座：`uer/gpt2-chinese-cluecorpussmall`（中文 GPT-2）
- 数据：电商评论文本，过滤长度 20~1024 字
- 训 1 epoch，loss `nn.CrossEntropyLoss`（自回归 next-token）
- 输出：`instructgpt/gpt2-sft/`

### 阶段 2：Reward Model

```bash
uv run python instructgpt/reward_model.py
```

- 结构：GPT-2 主干 + `nn.Linear(hidden, 1)` 奖励头
- 监督信号：评论 → 正/负向 → 1.0 / 0.0（**用情感标签近似人类偏好**，简化 InstructGPT 的成对比较数据）
- loss：仅在 reward token（追加在末尾的 eos）位置算 BCE
- 输出：`instructgpt/reward_model.pt`

### 阶段 3：PPO

```bash
uv run python instructgpt/ppo.py
```

核心流程：

1. **Rollout**：用 `gpt2-sft` 当前权重 π_θ 对 prompt 采样补全 y
2. **Reward**：
   - 步内逐 token：`r_t = -β · KL(π_θ ‖ π_ref)[t]`（β=0.2）
   - 末位一次性：`r_T += RewardModel(x, y)`（缩放到 (-1, 1)）
3. **Advantage**：GAE(γ=1.0, λ=0.95) + masked whitening
4. **Update**：4 个 PPO epoch × mini-batch，loss = `clip(ratio, 1±0.2) · A` + 0.1 · MSE(V, returns)

---

## 🎯 GRPO 路线（DeepSeek-R1 风格）

### 任务：Countdown

> 给定 3~4 个整数 + 一个目标数，用 `+ - * /` 拼一个等式等于目标数。
>
> 例：`numbers=[3, 5, 7, 12], target=21` → `<answer> (12 / 3) * 7 - 7 </answer>`

### 训练命令

```bash
# 把 Qwen2.5-3B-Instruct/ 和 Countdown-Tasks-3to4/ 放到 grpo/ 目录下
uv run python grpo/train.py
```

### 算法要点

| 项 | 设置 |
|---|---|
| 每 batch 问题数 | 32 |
| 每问题采样回答数 | 8（组大小） |
| 总 batch | 256 trajectory |
| 优势估计 | 组内 Z-score：`A_i = (R_i − mean(R)) / (std(R) + 1e-4)` |
| 奖励 | `0.1 × format_reward + answer_reward` |
| 优化器 | AdamW, lr=1e-5, betas=(0.9, 0.999) |
| 梯度裁剪 | clip_grad_norm = 1.0 |
| 模型 dtype | bf16 训练 + checkpoint_sequential |

GRPO 与 PPO 的关键差异：**省掉 critic**，用同题 N 条回答的均值作天然 baseline，显存 ≈ ½ PPO。

---

## 🔧 手写底层实现（`model_notes/`）

| 文件 | 内容 |
|---|---|
| `gpt_2.py` | 经典 Transformer：MHA、可学位置编码、LayerNorm、GELU、tie weights、朴素 KV Cache（prefill + decode 两阶段） |
| `qwen_3.py` | 现代化改进：RoPE 旋转位置编码、RMSNorm、GQA、SwiGLU、QK-Norm；手写不依赖 transformers 的 chat-template tokenizer |

两份代码**互相对照可以一眼看到 2019 → 2024 这五年大模型架构的演进取舍**。不在 RLHF 训练链路上，纯学习用。

---

## 📊 量化结果

> 训练运行后填入实测数据。

| 指标 | 数值 |
|---|---|
| Countdown 上 Qwen2.5-3B-Instruct 基线正确率（128 道测试题） | `[?]%` |
| GRPO 训练 N 步后正确率 | `[?]%` |
| GRPO 训练步数 | `[?]` |
| 训练 wall-clock 时间 / 单卡 GPU 型号 | `[?]` |

训练曲线（answer_reward / format_reward / loss / entropy）可通过 TensorBoard 查看：

```bash
uv run tensorboard --logdir grpo/logs/
```

---

## 📚 参考资料

- **InstructGPT**: [Training language models to follow instructions with human feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- **DeepSeek-R1**: [DeepSeek-R1: Incentivizing Reasoning Capability (2025)](https://arxiv.org/abs/2501.12948)
- **GRPO 原始论文**: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning (2024)](https://arxiv.org/abs/2402.03300)
- **PPO**: [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- **GAE**: [High-Dimensional Continuous Control Using Generalized Advantage Estimation (Schulman et al., 2015)](https://arxiv.org/abs/1506.02438)
- **trl 库**（PPO 实现参考）: [huggingface/trl](https://github.com/huggingface/trl)

---

## 📝 License

MIT
