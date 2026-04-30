# Copyright © 2026 Apple Inc.

import unittest

from mlx_lm.chat_templates import deepseek_v32


class TestDeepSeekV32Template(unittest.TestCase):

    def test_enable_thinking_true_emits_think_token(self):
        prompt = deepseek_v32.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=True,
            enable_thinking=True,
        )
        self.assertTrue(prompt.endswith("<｜Assistant｜><think>"))

    def test_enable_thinking_false_emits_close_think(self):
        prompt = deepseek_v32.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        self.assertTrue(prompt.endswith("<｜Assistant｜></think>"))

    def test_thinking_mode_wins_over_enable_thinking(self):
        prompt = deepseek_v32.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=True,
            enable_thinking=True,
            thinking_mode="chat",
        )
        self.assertTrue(prompt.endswith("<｜Assistant｜></think>"))

    def test_unknown_kwargs_are_filtered(self):
        prompt = deepseek_v32.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=True,
            enable_thinking=True,
            tokenize=False,
            return_tensors=None,
        )
        self.assertTrue(prompt.endswith("<｜Assistant｜><think>"))

    def test_no_generation_prompt_strips_assistant_marker(self):
        prompt_thinking = deepseek_v32.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=False,
            enable_thinking=True,
        )
        self.assertFalse(prompt_thinking.endswith("<｜Assistant｜><think>"))
        self.assertTrue(prompt_thinking.endswith("hi"))

        prompt_chat = deepseek_v32.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=False,
            enable_thinking=False,
        )
        self.assertFalse(prompt_chat.endswith("<｜Assistant｜></think>"))
        self.assertTrue(prompt_chat.endswith("hi"))


if __name__ == "__main__":
    unittest.main()
