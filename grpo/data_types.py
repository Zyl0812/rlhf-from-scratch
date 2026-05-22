"""GRPO 训练流程中使用的数据结构。

- `Episode`: 一条完整轨迹（问题 + 模型生成的一条回答 + 奖励信息），
  GRPO 的 rollout 阶段为每个问题采样 N 条 Episode 组成一个"组"。
- `MiniBatch`: 训练时一个 step 喂给模型的一批问题（不含回答），
  rollout 之后才会被展开成 N × batch_size 条 Episode。
"""

from dataclasses import dataclass


@dataclass
class Episode:
    """一个回合 = 一道问题 + 一条采样回答 + 该回答得到的奖励。

    GRPO 的 `normalize_rewards_per_group` 会按 `prefix`（即问题）
    把同组的多条 Episode 聚合，做组内优势标准化。
    """

    prefix: str  # 问题文本（完整 chat prompt，含 system / user / 助手前缀）
    text: str  # "问题 + 回答" 的完整拼接文本，仅用于日志/可视化
    prefix_token_ids: list[int]  # 问题部分的 input_ids（不含回答）
    prefix_tokens: list[str]  # 问题部分的 token 字符串列表
    generated_token_ids: list[int]  # 模型生成的回答 token id 列表（不含 prefix）
    is_finished: bool  # 回答是否在 max_gen_len 内自然结束（遇到 eos）
    reward: float  # 标量奖励；在 normalize 之后该字段被替换为"组内优势"
    reward_info: dict[str, float]  # 奖励的拆分（如 format_reward / answer_reward）


@dataclass
class MiniBatch:
    """每个训练 step 从 DataLoader 取到的一批问题。

    注意：这里只包含问题（prefix），不包含回答 —— 回答由 rollout 现采样得到。
    """

    prefix: list[str]  # 问题文本列表，长度 = batch_size
    prefix_tokens: list[list[str]]  # 每个问题的 token 字符串列表
    prefix_token_ids: list[list[int]]  # 每个问题的 input_ids，长度可不同
    numbers: list[list[int]]  # 每道 Countdown 题给定的数字列表（如 [3, 5, 7, 12]）
    target: list[int]  # 每道题需要凑出的目标数字
