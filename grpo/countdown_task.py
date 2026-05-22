"""Countdown 算术任务的数据集 + 规则奖励函数。

任务定义：给定 3~4 个整数和一个目标数，使用 +、-、*、/ 各拼一个等式，
使其结果等于目标数。例如：

    numbers = [3, 5, 7, 12], target = 21
    一个合法回答: <think>...</think>\\n<answer> (12 / 3) * 7 - 7 </answer>

奖励设计（DeepSeek-R1 风格的"规则奖励"）：
    format_reward  : 检查输出是否符合 <think>...</think>\\n<answer>...</answer> 格式
                     完全匹配 = 1.0，部分匹配（只有 <think> 或只有 <answer>）= 0.1 / 0.5
    answer_reward  : 答案表达式必须 (a) 只用允许字符；(b) 把给定数字各用一次；
                     (c) 求值后等于 target —— 三者都满足才给 1.0，否则 0.0
    总奖励         : 0.1 * format_reward + answer_reward
                     答案权重远高于格式，避免模型只学会套壳而不学解题

`eval(answer_content, {"__builtins__": None}, {})` 是受限的算术求值，
关掉所有内置函数防注入；但仍不要用本代码处理外部不可信输入。
"""

import re
from pathlib import Path
from typing import Any

import pandas as pd
from data_types import MiniBatch
from tokenizer import Tokenizer
from torch.utils.data import Dataset

SYSTEM_MESSAGE = "你是一个有用的助手。你首先在脑海中思考推理过程，然后为用户提供答案。"
# `{numbers}` 和 `{target}` 是占位符，构建训练数据时会被替换
USER_TEMPLATE = (
    "使用这些数字 {numbers}，创建一个等于 {target} 的等式。"
    "你可以使用基本算术运算（+、-、*、/），每个数字只能使用一次。"
    "在 <think> </think> 标签中展示你的解题过程。"
    "并在 <answer> </answer> 标签中返回最终答案，例如 <answer> (1 + 2) / 3 </answer>。"
)

RESPONSE_PROMPT = "让我一步步来解决这个问题。\n<think>"


class CountdownTasksDataset(Dataset):
    """准备训练数据集"""

    def __init__(
        self,
        tokenizer: Tokenizer,  # 分词器
        data_path: str,  # 数据集的路径
        split: str = "train",
        test_size: int = 100,
    ):
        data = pd.read_parquet(Path(data_path) / "data")
        # 索引 `test_size` 后面的数据用作测试数据
        self.data = data.iloc[:-test_size] if split == "train" else data.iloc[-test_size:]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx].to_dict()
        item.update(
            self.encode_prefix(
                item["nums"],  # 数字列表
                item["target"],  # 目标数字
            )
        )
        return item

    def encode_prefix(self, numbers: list[int], target: int):
        """Prefix 是模型 *真正的* 输入，也就是问题"""
        # 格式化对话模板
        user_message = USER_TEMPLATE.format(numbers=numbers, target=target)
        prefix = self.tokenizer.encode_chat_with_response_prompt(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            RESPONSE_PROMPT,
        )
        # 将问题切分
        tokens = self.tokenizer.tokenize(prefix)
        return {
            "prefix": prefix,  # 问题字符串
            "prefix_tokens": tokens.tokens,  # 问题切分后的字符串列表
            "prefix_token_ids": tokens.ids,  # input_ids
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> MiniBatch:
        """将数据整理到一个批次中"""
        numbers = [item["nums"] for item in batch]
        target = [item["target"] for item in batch]
        prefix = [item["prefix"] for item in batch]
        prefix_tokens = [item["prefix_tokens"] for item in batch]
        prefix_token_ids = [item["prefix_token_ids"] for item in batch]
        return MiniBatch(
            numbers=numbers,
            target=target,
            prefix=prefix,
            prefix_tokens=prefix_tokens,
            prefix_token_ids=prefix_token_ids,
        )


def format_reward_function(
    response: str,  # 模型的回答
    end_token: str | None = None,  # 结尾token
) -> float:
    """
    检查模型的回复是否符合格式 <think>...</think><answer>...</answer>
    """
    # 如果存在end token，则去掉
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]

    think_regex = r"<think>.*?<\/think>"
    answer_regex = r"<answer>.*?<\/answer>"
    full_format_regex = r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"

    think_match = re.search(think_regex, response, re.DOTALL)
    answer_match = re.search(answer_regex, response, re.DOTALL)
    full_format_match = re.match(full_format_regex, response, re.DOTALL)
    # 如果完全匹配，则给1分
    if full_format_match:
        return 1.0

    reward = 0.0
    # 如果有<think></think>标签对，则奖励加0.1分
    if think_match:
        reward += 0.1
    # 如果有<answer></answer>标签对，则奖励加0.5分
    if answer_match:
        reward += 0.5
    # 返回奖励
    return reward


def answer_reward_function(
    response: str,  # 模型给出的回答
    numbers: list[int] = None,  # 数字列表
    target: int = None,  # 目标数字
) -> float:
    """
    检查答案中：
    1. 是否使用了所有给的数字
    2. 每个数字是否使用了一次
    3. 答案中包含的表达式的求值结果是否等于目标数字
    """
    # 答案的正则表达式
    answer_regex = r"<answer>(.*?)<\/answer>"
    # 回答中是否有答案标签对
    answer_match = re.search(answer_regex, response, re.DOTALL)
    # 如果在回答中没有搜索到答案，那么给0分
    if not answer_match:
        return 0.0
    # 提取出答案的文本
    answer_content = answer_match.group(1)
    # 如果答案标签内没有东西，给0分
    if not answer_content:
        return 0.0
    # 如果答案标签中，除了表达式以外，还有其它内容，给0分
    allowed_chars = r"^[0-9+\-*/() ]+$"
    if not re.match(allowed_chars, answer_content):
        return 0.0

    # 检查答案中，每个数字是否只使用了一次
    used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
    if sorted(used_numbers) != sorted(numbers):
        return 0.0

    # 检查答案中包含的表达式的求值结果是否为目标数字
    try:
        result = eval(answer_content, {"__builtins__": None}, {})
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
    except Exception:
        # 表达式解析/求值失败（如除零、语法错误）都视为答错
        pass

    return 0.0


def reward_function(
    response: str,
    numbers: list[int] = None,
    target: int = None,
    end_token: str = None,
) -> dict[str, Any]:
    """Countdown Task 的奖励函数。

    总奖励 = 0.1 * 格式奖励 + 答案准确性奖励
    """
    format_reward = format_reward_function("<think>" + response, end_token)
    answer_reward = answer_reward_function(response, numbers, target)
    return {
        "reward": format_reward * 0.1 + answer_reward,
        "reward_info": {
            "format_reward": format_reward,
            "answer_reward": answer_reward,
        },
    }
