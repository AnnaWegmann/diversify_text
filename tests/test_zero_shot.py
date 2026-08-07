"""Tests for the zero_shot method: instruction filling and input checks."""

import unittest
from unittest.mock import MagicMock

from diversify_text.method.zero_shot import ZERO_SHOT_STYLE_BANK, ZeroShotMethod


class TestBank(unittest.TestCase):

    def test_each_style_has_exactly_one_instruction(self):
        for name, instructions in ZERO_SHOT_STYLE_BANK.items():
            self.assertEqual(len(instructions), 1, name)


class TestGenerate(unittest.TestCase):

    def test_instruction_with_and_without_placeholder(self):
        method = ZeroShotMethod()
        mock_model = MagicMock()
        # The mock echoes each prompt, so the output shows the prompts
        # the method built.
        mock_model.generate_text.side_effect = lambda prompts, **kw: list(prompts)
        method._model = mock_model

        # Explicit max_new_tokens skips the automatic token budget,
        # which would otherwise need the model's tokenizer.
        result = method.generate(
            ["my text"],
            {
                "quote": ["Repeat this: [DOCUMENT SEGMENT], please."],
                "pirate": ["Rewrite the text as a pirate."],
            },
            max_new_tokens=32,
        )
        self.assertEqual(
            result,
            [[
                "Repeat this: my text, please.",
                "Rewrite the text as a pirate."
                "\n\nOutput only the rewrite, nothing else.\nText: my text",
            ]],
        )

    def test_more_than_one_instruction_raises(self):
        method = ZeroShotMethod()
        with self.assertRaises(ValueError) as cm:
            method.generate(["text"], {"pirate": ["one", "two"]})
        self.assertEqual(
            str(cm.exception),
            "Style 'pirate' must have exactly one instruction, got 2.",
        )


if __name__ == "__main__":
    unittest.main()
