"""Tests if the prompt bank in prompts.py is valid (all templates are example-based)
    and prompt resolution logic."""

import unittest

from diversify_text.method.prompting.method import PromptingMethod
from diversify_text.method.prompting.prompts import (
    DEFAULT_PROMPT,
    PLACEHOLDER_STYLE_EXAMPLES,
    PLACEHOLDER_STYLE_NAME,
    PLACEHOLDER_TEXT,
    PROMPT_BANK,
)


class TestPromptBankValidity(unittest.TestCase):

    def test_all_bank_templates_contain_required_placeholders(self):
        for key, template in PROMPT_BANK.items():
            self.assertIn(
                PLACEHOLDER_TEXT,
                template,
                f"Prompt '{key}' is missing the {PLACEHOLDER_TEXT} placeholder.",
            )
            self.assertIn(
                PLACEHOLDER_STYLE_EXAMPLES,
                template,
                f"Prompt '{key}' is missing the {PLACEHOLDER_STYLE_EXAMPLES} placeholder.",
            )

    def test_default_prompt_is_valid_key(self):
        self.assertIn(DEFAULT_PROMPT, PROMPT_BANK)


class TestResolvePrompt(unittest.TestCase):

    def test_default_returns_default_prompt(self):
        key, template = PromptingMethod._resolve_prompt()
        self.assertEqual(key, DEFAULT_PROMPT)
        self.assertEqual(template, PROMPT_BANK[DEFAULT_PROMPT])

    # --- TEST SELECTING A PROMPT BY KEY ---

    def test_select_prompt_by_key(self):
        key, template = PromptingMethod._resolve_prompt("humanize_transfer")
        self.assertEqual(key, "humanize_transfer")
        self.assertEqual(template, PROMPT_BANK["humanize_transfer"])

    def test_unknown_key_raises_and_lists_available(self):
        with self.assertRaises(ValueError) as context_manager:
            PromptingMethod._resolve_prompt("nonexistent")
        message = str(context_manager.exception)
        self.assertIn("nonexistent", message)
        self.assertIn("style_transfer", message)

    # --- TEST CUSTOM TEMPLATES ---

    def test_custom_template_with_all_required_placeholders(self):
        custom = (
            "Examples: [STYLE EXAMPLES]\nRewrite this: [DOCUMENT SEGMENT]"
        )
        key, template = PromptingMethod._resolve_prompt(custom)
        self.assertEqual(key, "custom")
        self.assertEqual(template, custom)

    def test_custom_template_style_name_is_optional(self):
        custom = (
            "Style [STYLE NAME] examples: [STYLE EXAMPLES]\n"
            "Text: [DOCUMENT SEGMENT]"
        )
        key, _template = PromptingMethod._resolve_prompt(custom)
        self.assertEqual(key, "custom")

    def test_custom_template_missing_style_examples_raises(self):
        with self.assertRaises(ValueError) as context_manager:
            PromptingMethod._resolve_prompt("Rewrite: [DOCUMENT SEGMENT]")
        self.assertIn(PLACEHOLDER_STYLE_EXAMPLES, str(context_manager.exception))

    def test_custom_template_missing_document_raises(self):
        with self.assertRaises(ValueError) as context_manager:
            PromptingMethod._resolve_prompt("Examples: [STYLE EXAMPLES]")
        self.assertIn(PLACEHOLDER_TEXT, str(context_manager.exception))

    def test_custom_template_error_names_all_missing_placeholders(self):
        # Only [STYLE NAME] present → both required placeholders reported.
        with self.assertRaises(ValueError) as context_manager:
            PromptingMethod._resolve_prompt("Rewrite in [STYLE NAME] style.")
        message = str(context_manager.exception)
        self.assertIn(PLACEHOLDER_TEXT, message)
        self.assertIn(PLACEHOLDER_STYLE_EXAMPLES, message)


class TestFillTemplate(unittest.TestCase):

    def test_replaces_document_placeholder(self):
        method = PromptingMethod()
        result = method._fill_template(
            template="Rewrite: [DOCUMENT SEGMENT]",
            text="my text",
        )
        self.assertNotIn(PLACEHOLDER_TEXT, result)
        self.assertIn("my text", result)

    def test_replaces_all_placeholders(self):
        method = PromptingMethod()
        template = (
            "Style: [STYLE NAME]\n"
            "Examples: [STYLE EXAMPLES]\n"
            "Text: [DOCUMENT SEGMENT]"
        )
        result = method._fill_template(
            template=template,
            text="hello",
            style_idx=0,
            fs_style_examples={"informal_custom": ["example one", "example two"]},
            n_style_examples=2,
        )
        self.assertNotIn(PLACEHOLDER_TEXT, result)
        self.assertNotIn(PLACEHOLDER_STYLE_EXAMPLES, result)
        self.assertNotIn(PLACEHOLDER_STYLE_NAME, result)
        self.assertIn("hello", result)
        self.assertIn("example one", result)
        self.assertIn("informal", result)


if __name__ == "__main__":
    unittest.main()
