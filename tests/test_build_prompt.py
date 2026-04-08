"""Tests if the prompt bank in prompts.py is valid (e.g., few shot prompts allow for addition of style examples)
    and prompt resolution logic."""

import unittest

from diversify_text.method.prompting.method import PromptingMethod
from diversify_text.method.prompting.prompts import (
    DEFAULT_PROMPTS,
    FEW_SHOT_PROMPT_BANK,
    PLACEHOLDER_STYLE_EXAMPLES,
    PLACEHOLDER_STYLE_NAME,
    PLACEHOLDER_TEXT,
    PROMPT_BANK,
    ZS_PROMPT_BANK,
)


class TestPromptBankValidity(unittest.TestCase):

    def test_default_prompt_bank_templates_contain_placeholder(self):
        for key, template in ZS_PROMPT_BANK.items():
            self.assertIn(
                PLACEHOLDER_TEXT,
                template,
                f"Prompt '{key}' is missing the {PLACEHOLDER_TEXT} placeholder.",
            )

    def test_few_shot_prompt_bank_templates_contain_placeholders(self):
        for key, template in FEW_SHOT_PROMPT_BANK.items():
            self.assertIn(
                PLACEHOLDER_TEXT,
                template,
                f"Few-shot prompt '{key}' is missing the {PLACEHOLDER_TEXT} placeholder.",
            )
            self.assertIn(
                PLACEHOLDER_STYLE_EXAMPLES,
                template,
                f"Few-shot prompt '{key}' is missing the {PLACEHOLDER_STYLE_EXAMPLES} placeholder.",
            )
            self.assertIn(
                PLACEHOLDER_STYLE_NAME,
                template,
                f"Few-shot prompt '{key}' is missing the {PLACEHOLDER_STYLE_NAME} placeholder.",
            )

    def test_default_prompts_reference_valid_keys(self):
        for key in DEFAULT_PROMPTS:
            self.assertIn(key, ZS_PROMPT_BANK)


class TestResolvePrompts(unittest.TestCase):

    def test_default_returns_default_prompts(self):
        templates = PromptingMethod._resolve_prompts()
        self.assertEqual(len(templates), len(DEFAULT_PROMPTS))
        for t in templates:
            self.assertIn(PLACEHOLDER_TEXT, t)

    # --- TEST SELECTING PROMPT BY KEY ---
    def test_select_specific_prompt_keys(self):
        templates = PromptingMethod._resolve_prompts(
            prompt_keys=["wikipedia_paraphrase"]
        )
        self.assertEqual(len(templates), 1)
        self.assertEqual(templates[0], PROMPT_BANK["wikipedia_paraphrase"])

    def test_unknown_key_raises(self):
        with self.assertRaises(ValueError) as context_manager:
            PromptingMethod._resolve_prompts(prompt_keys=["nonexistent"])
        self.assertIn("nonexistent", str(context_manager.exception))

    # --- TEST SELECTING FEW-SHOT STYLE EXAMPLES BY KEYS ---

    def test_style_example_keys_without_prompt_keys_returns_few_shot(self):
        templates = PromptingMethod._resolve_prompts(
            style_example_keys=["informal_tinystyler"]
        )
        self.assertEqual(len(templates), 1)
        self.assertEqual(templates[0], FEW_SHOT_PROMPT_BANK["style_transfer"])

    def test_style_example_keys_with_few_shot_prompt_keys(self):
        templates = PromptingMethod._resolve_prompts(
            style_example_keys=["informal_tinystyler"],
            prompt_keys=["humanize_transfer"],
        )
        self.assertEqual(len(templates), 1)
        self.assertIn(PLACEHOLDER_STYLE_EXAMPLES, templates[0])

    def test_style_example_keys_with_zero_shot_prompt_keys_raises(self):
        with self.assertRaises(ValueError) as context_manager:
            PromptingMethod._resolve_prompts(
                style_example_keys=["informal_tinystyler"],
                prompt_keys=["wikipedia_paraphrase"],
            )
        self.assertIn(PLACEHOLDER_STYLE_EXAMPLES, str(context_manager.exception))

    # --- TEST CUSTOM PROMPT BANK ---

    def test_custom_bank(self):
        bank = {"custom": "Rewrite: [DOCUMENT SEGMENT]"}
        templates = PromptingMethod._resolve_prompts(prompt_bank=bank)
        self.assertEqual(templates, ["Rewrite: [DOCUMENT SEGMENT]"])

    def test_custom_bank_with_selection(self):
        bank = {"a": "prompt a [DOCUMENT SEGMENT]", "b": "prompt b [DOCUMENT SEGMENT]"}
        templates = PromptingMethod._resolve_prompts(prompt_bank=bank, prompt_keys=["b"])
        self.assertEqual(len(templates), 1)
        self.assertIn("prompt b", templates[0])


if __name__ == "__main__":
    unittest.main()
