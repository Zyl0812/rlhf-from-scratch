# 模型底层实现笔记

不参与 RLHF 训练链路，**纯学习用**：把 Transformer 各组件从零写一遍，对照不同年代/架构的取舍。

## 文件

| 文件 | 模型 | 关键特点 |
|---|---|---|
| `gpt_2.py` | GPT-2 (124M) 经典 | MHA、可学习位置编码、LayerNorm（带均值&偏置）、GELU、tie embeddings、朴素 KV Cache（prefill + decode 两阶段） |
| `qwen_3.py` | Qwen3 (0.6B) 现代 | RoPE 旋转位置编码、RMSNorm（无偏置）、GQA（K/V 头数 < Q 头数）、SwiGLU、可选 QK-Norm |

## 五年间的架构演进对照

| 组件 | GPT-2 (2019) | Qwen3 (2025) | 动机 |
|---|---|---|---|
| 位置编码 | 可学习 absolute embedding | RoPE | RoPE 可外推、零参数、相对位置感 |
| Norm | LayerNorm (mean + var + bias) | RMSNorm (仅 var) | 去掉一半计算，效果不掉 |
| Attention | MHA (n_kv = n_q) | GQA (n_kv ≪ n_q) | K/V cache 显存压力大幅降低 |
| 激活 | GELU | SwiGLU (gate · up) | SwiGLU 在 scale law 上更优 |
| QK 处理 | 直接点积 | QK-Norm（可选） | 大模型训练数值稳定 |
| 输出层 | weight tying | 可选 | 略 |

## 运行

```bash
# gpt_2.py：自带 main()，会跑一段文本生成 + 计时
uv run python model_notes/gpt_2.py

# qwen_3.py：需要先把 Qwen3-0.6B/tokenizer.json 放在工作目录
uv run python model_notes/qwen_3.py
```

## 与 `grpo/qwen2_model.py` 的区别

- `grpo/qwen2_model.py` 是 **训练 + 推理两用** 的生产级实现，要配合 SafeTensors 加载、KV Cache 复用、gradient checkpointing
- `model_notes/qwen_3.py` 是 **纯学习版**，逐组件展开写，可读性优先
