"""Tests for the PromptingMethod constructor, generation, and model backend dispatch."""

import unittest
from unittest.mock import MagicMock, patch

from diversify_text.method.prompting.method import PromptingMethod

class TestPromptingMethodConstructor(unittest.TestCase):

    def test_default_model(self):
        method = PromptingMethod()
        self.assertEqual(method.model_id, "HuggingFaceTB/SmolLM3-3B")

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
        )
        # .kwargs holds the keyword arguments of the generate_text call.
        call_kwargs = method._model.generate_text.call_args.kwargs
        self.assertEqual(call_kwargs["temperature"], 0.7)
        self.assertEqual(call_kwargs["top_p"], 0.9)
        # Mock tokenizer returns 3 tokens → max(10, min(3*2, 2048)) = 10.
        self.assertEqual(call_kwargs["max_new_tokens_per_prompt"], [10])

    def test_generate_uses_selected_prompt_key(self):
        method = self._make_method_with_mock_model(["out"])
        method.generate(
            ["text"],
            n=1,
            max_new_tokens=None,
            temperature=None,
            top_p=None,
            prompt="humanize_transfer",
        )
        # .args[0] is the first positional argument of the generate_text
        # call: the list of fully-filled prompt strings sent to the model.
        sent_prompts = method._model.generate_text.call_args.args[0]
        self.assertIn("machine-generated text", sent_prompts[0])

    def test_generate_uses_custom_prompt_template(self):
        method = self._make_method_with_mock_model(["out"])
        custom_prompt = (
            "Here are examples of the target style:\n[STYLE EXAMPLES]\n"
            "Rewrite the following text in that style: [DOCUMENT SEGMENT]"
        )
        method.generate(
            ["my input"],
            n=1,
            max_new_tokens=None,
            temperature=None,
            top_p=None,
            prompt=custom_prompt,
            custom_style_bank={"my_style": ["style example a"]},
        )
        # .args[0] is the first positional argument of the generate_text
        # call: the list of fully-filled prompt strings sent to the model.
        sent_prompts = method._model.generate_text.call_args.args[0]
        self.assertIn("my input", sent_prompts[0])
        self.assertIn("style example a", sent_prompts[0])

    def test_generate_rejects_prompt_without_style_examples(self):
        method = self._make_method_with_mock_model(["out"])
        with self.assertRaises(ValueError):
            method.generate(
                ["text"],
                n=1,
                max_new_tokens=None,
                temperature=None,
                top_p=None,
                prompt="Rewrite: [DOCUMENT SEGMENT]",
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
