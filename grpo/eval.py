"""Countdown 测试集上的正确率评估脚本。

两种使用方式：

1. **作为模块被 train.py 调用**（训练中每 N 步评估一次）：
       from eval import evaluate
       acc = evaluate(model, tokenizer, device, dtype, data_path="...")

2. **作为独立脚本运行**，评估基线模型或某个 checkpoint：
       # 评估未训练的 Qwen2.5-3B-Instruct（即基线）：
       python eval.py --model-path ./Qwen2.5-3B-Instruct/ \\
                      --data-path ./Countdown-Tasks-3to4/

       # 评估某个训练后的 checkpoint：
       python eval.py --model-path ./Qwen2.5-3B-Instruct/ \\
                      --data-path ./Countdown-Tasks-3to4/ \\
                      --ckpt ./ckpt/ckpt_000080.pt

`evaluate()` 的核心逻辑：在测试集 128 道题上各采样 1 条回答，
统计 `answer_reward == 1.0` 的比例（即"做对的占比"）。
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from countdown_task import CountdownTasksDataset, reward_function
from qwen2_model import Transformer
from tokenizer import Tokenizer
from torch.utils.data import DataLoader

from grpo import rollout

# 评估批次大小：训练时 batch=256，评估时把单条生成长度翻倍，所以 batch 减半
EVAL_BATCH_SIZE = 256 // 2
# 评估时单条生成长度（训练时是 1024，留更长是为了不让答案被截断）
EVAL_MAX_GEN_LEN = 1024 * 2


def evaluate(
    model: Transformer,
    tokenizer: Tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    data_path: str = "./Countdown-Tasks-3to4/",
    test_size: int = 128,
) -> float:
    """在测试集上跑一遍，返回答案正确率（0~1）。"""
    test_dataset = CountdownTasksDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        split="test",
        test_size=test_size,
    )
    generator = torch.Generator(device=device)
    dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=CountdownTasksDataset.collate_fn,
        generator=generator,
        batch_size=EVAL_BATCH_SIZE,
        drop_last=False,
    )
    success = []
    for batch in dataloader:
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=EVAL_MAX_GEN_LEN,
            num_answer_per_question=1,  # 评估时每题只采 1 条回答
            reward_function=reward_function,
            device=device,
            dtype=dtype,
        )
        # answer_reward = 1.0 表示答案完全正确；取均值就是正确率
        success.extend([episode.reward_info["answer_reward"] for episode in episodes])
    return float(np.mean(success))


def main():
    parser = argparse.ArgumentParser(description="评估 Qwen2.5 基线或某个训练 checkpoint")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./Qwen2.5-3B-Instruct/",
        help="预训练模型目录，必须含 config.json 与 tokenizer.json",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./Countdown-Tasks-3to4/",
        help="数据集目录，必须含 data/*.parquet",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="可选：训练后的 ckpt_*.pt 路径。不传则评估基线（未训练的模型）",
    )
    parser.add_argument("--test-size", type=int, default=128, help="测试集大小")
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.set_default_device(device)
    torch.random.manual_seed(1337)

    print(f"加载模型: {args.model_path}")
    tokenizer = Tokenizer(str(Path(args.model_path) / "tokenizer.json"))
    model = Transformer.from_pretrained(args.model_path, device=device)

    if args.ckpt is not None:
        print(f"加载 checkpoint: {args.ckpt}")
        state_dict = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state_dict, strict=True)
    else:
        print("评估基线（未训练的预训练模型）")

    model.eval()
    print(f"在 {args.test_size} 道测试题上评估...")
    acc = evaluate(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        data_path=args.data_path,
        test_size=args.test_size,
    )
    print(f"\n答案正确率: {acc:.4f}  ({acc * 100:.2f}%)")


if __name__ == "__main__":
    main()
