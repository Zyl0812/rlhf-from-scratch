"""Qwen2.5 分词器的轻量封装。

只做两件 HuggingFace tokenizer 之外的事：
1. 用 jinja2 渲染 `tokenizer_config.json` 里的 `chat_template`，把
   `[{role, content}, ...]` 渲染成 Qwen 的 `<|im_start|>...<|im_end|>` 序列；
2. 把 chat 模板拼接好的字符串再追加一段 "response prompt"（例如
   "让我一步步来解决这个问题。\\n<think>"），让模型从一个固定开头继续生成。

这样做的好处是把 chat 格式和 response 引导词与训练/推理代码解耦：
GRPO 只关心拿到一段 `prefix` 字符串，至于这段字符串怎么拼是 tokenizer 的事。
"""

import json
from pathlib import Path

from jinja2 import Environment
from tokenizers import Encoding
from tokenizers import Tokenizer as TokenizerBase


class Tokenizer:
    """带 chat template 支持的 Qwen2 分词器（基于 `tokenizers` 库 + jinja2 模板）。"""

    def __init__(self, tokenizer_path: str):
        super().__init__()
        # tokenizer_config.json 与 tokenizer.json 同目录，里面有 chat_template/eos/pad 等元信息
        tokenizer_config_path = Path(tokenizer_path).parent / "tokenizer_config.json"
        self.tokenizer_config = json.load(open(tokenizer_config_path))
        self.tokenizer = TokenizerBase.from_file(tokenizer_path)
        # 把 jinja2 模板预编译一次，后续渲染消息直接复用
        self.chat_template = Environment().from_string(self.tokenizer_config["chat_template"])
        self.eos_token = self.tokenizer_config["eos_token"]
        self.eos_token_id = self.tokenizer.token_to_id(self.eos_token)
        self.pad_token = self.tokenizer_config["pad_token"]
        self.pad_token_id = self.tokenizer.token_to_id(self.pad_token)

    def encode_chat(self, messages: list[dict[str, str]]) -> str:
        """把 `[{role, content}, ...]` 渲染成符合 Qwen chat 协议的字符串。

        `add_generation_prompt=True` 会在末尾补上 `<|im_start|>assistant\\n`，
        告诉模型接下来要由 assistant 角色继续生成。
        """
        return self.chat_template.render(messages=messages, add_generation_prompt=True)

    def encode_chat_with_response_prompt(self, messages: list[dict[str, str]], prompt: str) -> str:
        """在 chat 模板末尾再拼一段 `prompt`，作为模型生成的起手内容。"""
        return self.encode_chat(messages) + prompt

    def tokenize(self, text: str) -> Encoding:
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids: list[int]) -> str:
        # 保留 special tokens 是为了在日志里能直接看到 `<|im_end|>` 等标记
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
