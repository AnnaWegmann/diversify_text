"""Tests for TinyStyler's style-dict generation (the future generate(), #18)."""

import unittest
from unittest.mock import MagicMock

from diversify_text.method.tinystyler import TinyStylerMethod


def _method_with_mock_model() -> TinyStylerMethod:
    """TinyStylerMethod whose model echoes '<text>|<first style example>'."""
    method = TinyStylerMethod()
    mock_model = MagicMock()
    mock_model.transfer.side_effect = (
        lambda texts, style_examples, **kw: [
            f"{t}|{style_examples[0]}" for t in texts
        ]
    )
    mock_model._tokenizer = lambda texts, **kw: {
        "input_ids": [[1, 2, 3]] * len(texts)
    }
    method._model = mock_model
    return method


class TestGenerateFromStyleDict(unittest.TestCase):

    def test_one_output_per_style_per_text_in_dict_order(self):
        method = _method_with_mock_model()
        result = method._generate_from_style_dict(
            ["text a", "text b"],
            {"casual": ["hey there"], "formal": ["Good day."]},
        )
        self.assertEqual(
            result,
            [
                ["text a|hey there", "text a|Good day."],
                ["text b|hey there", "text b|Good day."],
            ],
        )


if __name__ == "__main__":
    unittest.main()
