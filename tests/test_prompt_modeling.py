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

    def test_generate_applies_defaults_for_none_params(self):
        method = self._make_method_with_mock_model(["out"])
        method.generate(["text"], {"casual": ["hey there"]})
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
            {"casual": ["hey there"]},
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
            {"my_style": ["style example a"]},
            prompt=custom_prompt,
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
                {"casual": ["hey there"]},
                prompt="Rewrite: [DOCUMENT SEGMENT]",
            )


class TestGenerateFromStyleDict(unittest.TestCase):

    def test_one_output_per_style_in_dict_order(self):
        method = PromptingMethod()
        mock_model = MagicMock()

        def fake_generate(prompts, **kwargs):
            # Answer each prompt with the style example it contains, so
            # the output directly shows which style produced it.
            return [
                "casual reply" if "hey there" in p else "formal reply"
                for p in prompts
            ]

        mock_model.generate_text.side_effect = fake_generate
        method._model = mock_model

        # Explicit max_new_tokens skips the automatic token budget,
        # which would otherwise need the model's tokenizer.
        result = method.generate(
            ["my text"],
            {"casual": ["hey there"], "formal": ["Good day."]},
            max_new_tokens=32,
        )

        # One text, two styles → one output per style, in dict order.
        self.assertEqual(result, [["casual reply", "formal reply"]])


class TestPromptingModelLoad(unittest.TestCase):

    @patch("diversify_text.method.llm.PromptingModel._load_transformers")
    def test_load_calls_transformers(self, mock_load_tf):
        from diversify_text.method.llm import PromptingModel

        model = PromptingModel(model_id="test-model", device="cpu")
        model.load()
        mock_load_tf.assert_called_once()


if __name__ == "__main__":
    unittest.main()
