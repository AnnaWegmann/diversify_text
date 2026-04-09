"""Tests for the PromptingMethod constructor, generation, and model backend dispatch."""

import unittest
from unittest.mock import MagicMock, patch

from diversify_text.method.prompting.method import PromptingMethod

class TestPromptingMethodConstructor(unittest.TestCase):

    def test_default_model(self):
        method = PromptingMethod()
        self.assertEqual(method.model_id, "HuggingFaceTB/SmolLM2-1.7B-Instruct")

    def test_custom_model(self):
        method = PromptingMethod(model="mistralai/Mistral-7B-Instruct-v0.3")
        self.assertEqual(method.model_id, "mistralai/Mistral-7B-Instruct-v0.3")

    def test_device_stored(self):
        method = PromptingMethod(device="cpu")
        self.assertEqual(method.device, "cpu")


class TestPromptingMethodGenerate(unittest.TestCase):

    def _make_method_with_mock_model(self, responses: list[str]) -> PromptingMethod:
        """Create a PromptingMethod with a mocked model that returns *responses*."""
        method = PromptingMethod()
        mock_model = MagicMock()
        mock_model.generate_text.return_value = responses
        # Mock tokenizer: returns fake token IDs (3 tokens per text)
        # so max_new_tokens can be computed.
        mock_tokenizer = MagicMock()
        mock_tokenizer.side_effect = lambda texts, **kw: {
            "input_ids": [[1, 2, 3]] * len(texts)
        }
        mock_model._tokenizer = mock_tokenizer
        method._model = mock_model
        return method

    def test_generate_multiple_texts(self):
        method = self._make_method_with_mock_model(["para a", "para b"])
        result = method.generate(
            ["text a", "text b"],
            n=1,
            max_new_tokens=None,
            temperature=None,
            top_p=None,
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], "para a")
        self.assertEqual(result[1][0], "para b")

    def test_generate_multiple_styles(self):
        # n=3 with 1 text → 3 prompts in a single generate_text call.
        method = self._make_method_with_mock_model(["para1", "para2", "para3"])
        result = method.generate(
            ["hello"],
            n=3,
            max_new_tokens=None,
            temperature=None,
            top_p=None,
        )
        self.assertEqual(len(result[0]), 3)
        self.assertEqual(result[0], ["para1", "para2", "para3"])
        method._model.generate_text.assert_called_once()

    def test_generate_applies_defaults_for_none_params(self):
        method = self._make_method_with_mock_model(["out"])
        method.generate(
            ["text"],
            n=1,
            max_new_tokens=None,
            temperature=None,
            top_p=None,
            # Use a non-finephrase prompt so there's no bonus.
            prompt_keys=["wikipedia_paraphrase"],
        )
        call_kwargs = method._model.generate_text.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.7)
        self.assertEqual(call_kwargs["top_p"], 0.9)
        # Non-finephrase prompt, no bonus.
        # Mock tokenizer returns 3 tokens → max(10, min(3*2, 2048)) = 10.
        self.assertEqual(call_kwargs["max_new_tokens_per_prompt"], [10])

    def test_finephrase_bonus_only_applies_to_finephrase_prompts(self):
        # n=5 with 1 text, explicitly cycling 5 prompts:
        # 0: humanize (no bonus), 1: wikipedia (no bonus),
        # 2: finephrase_faq (bonus), 3: finephrase_table (bonus),
        # 4: finephrase_narrative (bonus).
        # Base = max(10, min(3*2, 2048)) = 10, bonus = 50.
        method = self._make_method_with_mock_model(
            ["out1", "out2", "out3", "out4", "out5"]
        )
        method.generate(
            ["text"],
            n=5,
            max_new_tokens=None,
            temperature=None,
            top_p=None,
            prompt_keys=[
                "humanize_llm-as-coauthor_original",
                "wikipedia_paraphrase",
                "finephrase_faq",
                "finephrase_table",
                "finephrase_narrative",
            ],
        )
        call_kwargs = method._model.generate_text.call_args[1]
        self.assertEqual(
            call_kwargs["max_new_tokens_per_prompt"],
            [10, 10, 60, 60, 60],
        )


class TestPromptingModelLoad(unittest.TestCase):

    @patch("diversify_text.method.prompting.model.PromptingModel._load_transformers")
    def test_load_calls_transformers(self, mock_load_tf):
        from diversify_text.method.prompting.model import PromptingModel

        model = PromptingModel(model_id="test-model", device="cpu")
        model.load()
        mock_load_tf.assert_called_once()


if __name__ == "__main__":
    unittest.main()
