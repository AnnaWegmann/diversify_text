"""Tests for the zero_shot method: instruction filling and input checks."""

import unittest
from unittest.mock import MagicMock

from diversify_text.method.zero_shot import ZERO_SHOT_STYLE_BANK, ZeroShotMethod


class TestBank(unittest.TestCase):

    def test_each_style_has_exactly_one_instruction(self):
        for name, instructions in ZERO_SHOT_STYLE_BANK.items():
            self.assertEqual(len(instructions), 1, name)


class TestFillInstruction(unittest.TestCase):

    def test_placeholder_is_replaced(self):
        filled = ZeroShotMethod._fill_instruction(
            "Repeat this: [DOCUMENT SEGMENT], please.", "my text"
        )
        self.assertEqual(filled, "Repeat this: my text, please.")

    def test_text_is_appended_when_no_placeholder(self):
        filled = ZeroShotMethod._fill_instruction(
            "Rewrite the text as a pirate.", "my text"
        )
        self.assertEqual(
            filled,
            "Rewrite the text as a pirate."
            "\n\nOutput only the rewrite, nothing else.\nText: my text",
        )


class TestGenerate(unittest.TestCase):

    def test_one_output_per_style_in_dict_order(self):
        method = ZeroShotMethod()
        mock_model = MagicMock()

        def fake_generate(prompts, **kwargs):
            # Answer each prompt with a reply named after the style
            # whose instruction it contains, so the output directly
            # shows which style produced it.
            return [
                "formal reply" if "more formal" in p else "pirate reply"
                for p in prompts
            ]

        mock_model.generate_text.side_effect = fake_generate
        method._model = mock_model

        # Explicit max_new_tokens skips the automatic token budget,
        # which would otherwise need the model's tokenizer.
        result = method.generate(
            ["my text"],
            {
                "formal": ["Rewrite the text to be more formal."],
                "pirate": ["Rewrite the text as a pirate."],
            },
            max_new_tokens=32,
        )

        # One text, two styles → one output per style, in dict order.
        self.assertEqual(result, [["formal reply", "pirate reply"]])

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
