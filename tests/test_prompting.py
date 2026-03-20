"""Tests for the prompting diversification method."""

import unittest
from unittest.mock import MagicMock, patch

from diversify_text.method import DEFAULT_METHOD_REGISTRY
from diversify_text.method.prompting.method import PromptingMethod
from diversify_text.method.prompting.prompts import DEFAULT_PROMPT_BANK, DEFAULT_PROMPTS, PLACEHOLDER_TEXT


class TestPromptBank(unittest.TestCase):

    def test_default_prompt_bank_templates_contain_placeholder(self):
        for key, template in DEFAULT_PROMPT_BANK.items():
            self.assertIn(
                PLACEHOLDER_TEXT,
                template,
                f"Prompt '{key}' is missing the {PLACEHOLDER_TEXT} placeholder.",
            )

    def test_default_prompts_reference_valid_keys(self):
        for key in DEFAULT_PROMPTS:
            self.assertIn(key, DEFAULT_PROMPT_BANK)


class TestResolvePrompts(unittest.TestCase):

    def test_default_returns_default_prompts(self):
        templates = PromptingMethod._resolve_prompts()
        self.assertEqual(len(templates), len(DEFAULT_PROMPTS))
        for t in templates:
            self.assertIn(PLACEHOLDER_TEXT, t)

    def test_custom_bank(self):
        bank = {"custom": "Rewrite: [DOCUMENT SEGMENT]"}
        templates = PromptingMethod._resolve_prompts(prompt_bank=bank)
        self.assertEqual(templates, ["Rewrite: [DOCUMENT SEGMENT]"])

    def test_select_specific_prompts(self):
        templates = PromptingMethod._resolve_prompts(
            prompts=["wikipedia_paraphrase"]
        )
        self.assertEqual(len(templates), 1)
        self.assertEqual(templates[0], DEFAULT_PROMPT_BANK["wikipedia_paraphrase"])

    def test_unknown_key_raises(self):
        with self.assertRaises(ValueError) as ctx:
            PromptingMethod._resolve_prompts(prompts=["nonexistent"])
        self.assertIn("nonexistent", str(ctx.exception))

    def test_custom_bank_with_selection(self):
        bank = {"a": "prompt a [DOCUMENT SEGMENT]", "b": "prompt b [DOCUMENT SEGMENT]"}
        templates = PromptingMethod._resolve_prompts(prompt_bank=bank, prompts=["b"])
        self.assertEqual(len(templates), 1)
        self.assertIn("prompt b", templates[0])


class TestPromptingMethodRegistration(unittest.TestCase):

    def test_prompting_is_registered(self):
        self.assertIn("prompting", DEFAULT_METHOD_REGISTRY)

    def test_registry_resolves_to_prompting_method(self):
        cls = DEFAULT_METHOD_REGISTRY.get("prompting")
        self.assertIs(cls, PromptingMethod)


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
        # Mock tokenizer: returns fake token IDs so max_new_tokens can be computed.
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]]}
        mock_model._tokenizer = mock_tokenizer
        method._model = mock_model
        return method

    def test_generate_single_text_single_style(self):
        method = self._make_method_with_mock_model(["paraphrased text"])
        result = method.generate(
            ["Hello world"],
            n=1,
            max_new_tokens=None,
            temperature=None,
            top_p=None,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(result[0][0], "paraphrased text")

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
        method = PromptingMethod()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]]}
        mock_model._tokenizer = mock_tokenizer
        mock_model.generate_text.side_effect = [
            ["para1"],
            ["para2"],
            ["para3"],
        ]
        method._model = mock_model

        result = method.generate(
            ["hello"],
            n=3,
            max_new_tokens=None,
            temperature=None,
            top_p=None,
        )
        self.assertEqual(len(result[0]), 3)
        self.assertEqual(mock_model.generate_text.call_count, 3)

    def test_generate_passes_placeholder_replaced_prompt(self):
        method = self._make_method_with_mock_model(["output"])
        method.generate(
            ["my text"],
            n=1,
            max_new_tokens=None,
            temperature=None,
            top_p=None,
        )
        call_args = method._model.generate_text.call_args
        prompts_sent = call_args[0][0]
        self.assertNotIn(PLACEHOLDER_TEXT, prompts_sent[0])
        self.assertIn("my text", prompts_sent[0])

    def test_generate_strips_whitespace(self):
        method = self._make_method_with_mock_model(["  output with spaces  "])
        result = method.generate(
            ["text"],
            n=1,
            max_new_tokens=None,
            temperature=None,
            top_p=None,
        )
        self.assertEqual(result[0][0], "output with spaces")

    def test_generate_with_custom_prompt_bank(self):
        method = self._make_method_with_mock_model(["result"])
        method.generate(
            ["hello"],
            n=1,
            max_new_tokens=None,
            temperature=None,
            top_p=None,
            prompt_bank={"custom": "Custom prompt: [DOCUMENT SEGMENT]"},
        )
        call_args = method._model.generate_text.call_args
        prompts_sent = call_args[0][0]
        self.assertIn("Custom prompt: hello", prompts_sent[0])

    def test_generate_applies_defaults_for_none_params(self):
        method = self._make_method_with_mock_model(["out"])
        method.generate(
            ["text"],
            n=1,
            max_new_tokens=None,
            temperature=None,
            top_p=None,
        )
        call_kwargs = method._model.generate_text.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.7)
        self.assertEqual(call_kwargs["top_p"], 0.9)
        # Mock tokenizer returns 3 tokens → max(10, min(3*2, 2048)) = 10.
        self.assertEqual(call_kwargs["max_new_tokens"], 10)


class TestPromptingModelLoad(unittest.TestCase):

    @patch("diversify_text.method.prompting.model.PromptingModel._load_transformers")
    def test_load_calls_transformers(self, mock_load_tf):
        from diversify_text.method.prompting.model import PromptingModel

        model = PromptingModel(model_id="test-model", device="cpu")
        model.load()
        mock_load_tf.assert_called_once()


if __name__ == "__main__":
    unittest.main()
