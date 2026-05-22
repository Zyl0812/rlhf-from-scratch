# AutoDL GRPO 训练 + 评测指南

> 在 [AutoDL](https://www.autodl.com/) 上以 **¥50 以内** 的预算，跑完 Qwen2.5-3B 在 Countdown 任务上的"基线评估 → GRPO 训练 → 训练后评估"完整链路，拿到一对真实的"基线正确率 → 训练后正确率"数据。

## 0. 预算与时长估算

按 **A100-80G**（AutoDL 上约 ¥6-10/h）+ 80 步训练 计算：

| 阶段 | 预计时长 | 预计费用 |
|---|---|---|
| 环境准备 + ModelScope 下载（模型 ~6GB + 数据 ~25MB） | 20-30 分钟 | ¥3-5 |
| 基线评估（128 道测试题，每题采样 1 条） | 10-15 分钟 | ¥2-3 |
| GRPO 训练 80 步（每步 ≈ 90-120s） | 2-3 小时 | ¥15-30 |
| 训练后评估 | 10-15 分钟 | ¥2-3 |
| 下载 ckpt + 关机缓冲 | 5-10 分钟 | ¥1-2 |
| **合计** | **≈ 3-4 小时** | **¥25-45** |

⚠️ 务必启用 AutoDL 的「**无卡模式**」做下载与上传 —— 这时 GPU 不计费，CPU + 带宽不限速，能省一大笔。下载完再切回带卡模式。

---

## 1. 创建实例

### 1.1 选 GPU

| 推荐顺序 | GPU | 价格（参考） | 备注 |
|---|---|---|---|
| ★ | **A100-80G PCIe** | ¥6-10/h | 训练 + 推理一气呵成，最稳 |
|  | A100-40G SXM | ¥4-6/h | 显存紧张，需要把 `NUM_QUESTIONS_PER_BATCH` 从 32 降到 16 |
|  | H100 80G | ¥15-25/h | 比 A100 快但贵，预算 ¥50 内不划算 |
|  | RTX 4090 24G | ¥2-3/h | **跑不动 3B GRPO**，仅用于基线评估或 1.5B 模型 |

预算 ¥50 内 → 选 **A100-80G**，找最便宜的可用区即可。

### 1.2 选镜像

进入 **公共镜像 → PyTorch** 选最新的：
- 框架：`PyTorch 2.5.1` 或更新
- Python：`3.12`
- CUDA：`12.4` 或 `12.8`

> ⚠️ 本仓库要求 Python 3.12.12，如镜像是 3.10/3.11，需要在容器里另装 Python，复杂；优先选 3.12 镜像。

### 1.3 选磁盘

- **系统盘** 50G（够用）
- **数据盘** `/root/autodl-tmp/` 至少 30G（放 Qwen 模型 ~6GB + checkpoint ~6GB × 1 + 数据 + log）

---

## 2. 进实例

### 2.1 SSH 连接

AutoDL 控制台复制登录命令，本地终端：

```bash
ssh -p <port> root@<host>
# 输入密码
```

或在 AutoDL 网页版直接打开 JupyterLab 用 Terminal。

### 2.2 开启学术加速（仅限国内服）

```bash
source /etc/network_turbo
```

> 这一步只为 git clone GitHub 加速。下载模型/数据走 ModelScope，**不需要**学术加速。

---

## 3. 拉代码

```bash
cd /root/autodl-tmp
git clone https://github.com/Zyl0812/rlhf-from-scratch.git
cd rlhf-from-scratch
```

---

## 4. 装依赖

### 4.1 装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 4.2 同步项目依赖

```bash
cd /root/autodl-tmp/rlhf-from-scratch
uv sync
```

> uv 第一次会装 torch (cu128 wheel 约 2.5GB)，约 5-10 分钟。

### 4.3 装 modelscope（用于下载模型 + 数据）

```bash
uv pip install modelscope
```

---

## 5. 下载模型 + 数据（用 ModelScope）

### 5.1 切到无卡模式

> AutoDL 控制台 → 实例操作 → 「**关机**」→ 「**无卡模式开机**」。下载阶段 GPU 不计费。

### 5.2 下 Qwen2.5-3B-Instruct

```bash
cd /root/autodl-tmp/rlhf-from-scratch/grpo
uv run modelscope download --model Qwen/Qwen2.5-3B-Instruct --local_dir ./Qwen2.5-3B-Instruct
```

下载完成后目录结构：
```
grpo/Qwen2.5-3B-Instruct/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
└── ...
```

### 5.3 下 Countdown-Tasks-3to4 数据集

ModelScope 没有这个数据集 → 用 `huggingface_hub` 走 hf-mirror：

```bash
uv pip install huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
uv run huggingface-cli download \
    --repo-type dataset \
    Jiayi-Pan/Countdown-Tasks-3to4 \
    --local-dir ./Countdown-Tasks-3to4
```

下完后目录：
```
grpo/Countdown-Tasks-3to4/
└── data/
    └── train-00000-of-00001.parquet  (~25MB)
```

### 5.4 切回带卡模式

控制台「关机」→ 「**开机**」（这次正常带 GPU），重新 SSH。

---

## 6. 基线评估（最先跑！）

> 这一步必须放在训练之前 —— 训练会修改模型权重。

```bash
cd /root/autodl-tmp/rlhf-from-scratch/grpo
uv run python eval.py \
    --model-path ./Qwen2.5-3B-Instruct/ \
    --data-path ./Countdown-Tasks-3to4/ \
    --test-size 128
```

期待输出：
```
加载模型: ./Qwen2.5-3B-Instruct/
评估基线（未训练的预训练模型）
在 128 道测试题上评估...
答案正确率: 0.1250  (12.50%)
```

**把这个数字记下来 → 简历里的"基线 [?]%"。**

---

## 7. GRPO 训练

### 7.1 后台跑（防止 SSH 断连导致中断）

```bash
cd /root/autodl-tmp/rlhf-from-scratch/grpo
nohup uv run python train.py > train.log 2>&1 &
echo "PID = $!"
```

记下 PID，万一要 kill：`kill $PID`。

### 7.2 监控进度

实时看日志：
```bash
tail -f train.log
```

每行长这样：
```
步骤 1, 平均奖励: 0.12, 计算正确率: 0.05, 梯度裁剪: 0.45, 时长: 95.32, ...
评估数据集回答正确率: 0.13                                       <-- 每 10 步
```

### 7.3 训多少步够 ¥50 预算

- 每步 ≈ 90-120s（rollout 占大头）；每 10 步插一次 evaluate ≈ +60s
- 80 步预计 2-3 小时
- 训练循环代码本身没有"训 N 步就停"的开关 —— **手动盯着 step 数，到 80 步就 Ctrl+C**（如果是 nohup 就 `kill $PID`）

> 想自动停？编辑 `train.py:81` 在 `for step, batch in enumerate(train_dataloader, start=1):` 里加：
> ```python
> if step > 80:
>     break
> ```

### 7.4 训练中的 TensorBoard（可选）

新开一个终端：
```bash
cd /root/autodl-tmp/rlhf-from-scratch/grpo
uv run tensorboard --logdir logs/ --port 6006 --bind_all
```

AutoDL 控制台 → 「**自定义服务**」→ 端口 6006 → 一键代理到本地浏览器。

---

## 8. 保存 checkpoint

`train.py` 默认每 **100 步** 保存一次到 `ckpt/ckpt_000100.pt`。

如果你只训 80 步就停了，**checkpoint 不会自动保存** —— 需要在停止前手动改一下。最简方案：编辑 `train.py:158` 把 `step % 100` 改成 `step % 20`，这样训练中每 20 步保存一次，停止时手头总能有一个最近的 ckpt。

或者训练前直接改 `158` 行：
```python
if step % 20 == 0:  # 改为 20
```

---

## 9. 训练后评估

```bash
cd /root/autodl-tmp/rlhf-from-scratch/grpo
ls ckpt/   # 找到最新的 ckpt 文件名

uv run python eval.py \
    --model-path ./Qwen2.5-3B-Instruct/ \
    --data-path ./Countdown-Tasks-3to4/ \
    --ckpt ./ckpt/ckpt_000080.pt \
    --test-size 128
```

期待输出：
```
加载模型: ./Qwen2.5-3B-Instruct/
加载 checkpoint: ./ckpt/ckpt_000080.pt
在 128 道测试题上评估...
答案正确率: 0.5234  (52.34%)
```

**把这个数字记下来 → 简历里的"训练后 [?]%"。**

---

## 10. 下载产物

### 10.1 训练曲线（TensorBoard log）

```bash
# 本地终端：
scp -P <port> -r root@<host>:/root/autodl-tmp/rlhf-from-scratch/grpo/logs ./
```

或者用 AutoDL 自带的「JupyterLab 文件管理器」直接下载。

### 10.2 train.log（实测日志，简历可用作截图素材）

```bash
scp -P <port> root@<host>:/root/autodl-tmp/rlhf-from-scratch/grpo/train.log ./
```

### 10.3 checkpoint（可选，~6GB）

体积大，传回本地意义不大；除非你想本地再 evaluate 一次。

---

## 11. 关机！！！

**这是最容易忘记的一步**，关不关机几小时下来差几十块。

AutoDL 控制台 → 实例操作 → **关机**（不是「重启」，关机才不计费）。

如果短期不再用，建议「**释放实例**」 —— 实例会被删除（系统盘清空），但 `/root/autodl-tmp/` 数据盘保留 7 天（基础会员）。下次用时新建一个实例再挂载同一块数据盘。

---

## 附：填回简历的指标

跑完后，把以下两组数字填回根目录 [`README.md`](../README.md) 的「📊 量化结果」表格：

```markdown
| Countdown 上 Qwen2.5-3B-Instruct 基线正确率（128 道测试题） | 12.5% |
| GRPO 训练 80 步后正确率                                     | 52.3% |
| GRPO 训练步数                                               | 80    |
| 训练 wall-clock 时间 / 单卡 GPU 型号                        | 约 2.5h / A100-80G |
```

简历那条空缺也可以同步填上：

> Countdown 任务上 GRPO 答案正确率从基线 **12.5%** 提升至 **52.3%**

---

## 附：常见问题

**Q1：OOM（CUDA out of memory）了怎么办？**

修改 `train.py` 中 `main()` 里的两个常量：

```python
NUM_QUESTIONS_PER_BATCH = 16  # 原 32 → 16
# BATCH_SIZE = NUM_QUESTIONS_PER_BATCH * NUM_ANSWERS_PER_QUESTION = 128
```

显存能省一半，但每步训得更少，可能要训更多步才看到效果。

**Q2：rollout 很慢，每步 200s+，正常吗？**

KV Cache 没启用时会慢得多。看 `grpo/grpo.py:rollout` 里 `model.init_kv_cache(...)` 是否被正确调用。也可能是 max_gen_len 过大 —— 训练时是 1024，评估时是 2048，评估那一步会比训练步慢。

**Q3：训练 loss / 奖励完全不动？**

检查这些点：
- 模型是否真的加载了权重（`from_pretrained` 失败会沉默地用随机权重）
- 数据集是否正确（看一眼 `train.log` 里 `text_0` 输出，应该能看到完整的 Countdown 问题）
- 学习率（`train.py` 里 `lr=1e-5`，太大会爆训练，可降到 `5e-6`）

**Q4：忘了开「无卡模式」，下载花了 30 分钟，多花了多少钱？**

按 A100-80G ¥8/h 算，30 min ≈ ¥4。下次记得开无卡模式就好。

**Q5：训练中网络断了怎么办？**

`nohup` 会让训练继续跑（不依赖 SSH 会话）。重连后 `tail -f train.log` 接着看。

如果实例被强制关机了，从最近一个 ckpt 续跑：需要手动改 `train.py` 加载 `state_dict` —— 当前代码不支持续训，这是已知简化。
