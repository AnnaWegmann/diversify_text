"""Tests for the method registry."""

import unittest

from diversify_text.method import DEFAULT_METHOD_REGISTRY
from diversify_text.method.prompting import PromptingMethod
from diversify_text.method.tinystyler import TinyStylerMethod


class TestMethodRegistration(unittest.TestCase):

    def test_prompting_is_registered(self):
        self.assertIn("prompting", DEFAULT_METHOD_REGISTRY)

    def test_prompting_resolves_to_correct_class(self):
        cls = DEFAULT_METHOD_REGISTRY.get("prompting")
        self.assertIs(cls, PromptingMethod)

    def test_tinystyler_is_registered(self):
        self.assertIn("tinystyler", DEFAULT_METHOD_REGISTRY)

    def test_tinystyler_resolves_to_correct_class(self):
        cls = DEFAULT_METHOD_REGISTRY.get("tinystyler")
        self.assertIs(cls, TinyStylerMethod)

    def test_unknown_method_raises(self):
        with self.assertRaises(KeyError):
            DEFAULT_METHOD_REGISTRY.get("nonexistent")


if __name__ == "__main__":
    unittest.main()
