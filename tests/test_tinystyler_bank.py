"""Tests for TinyStyler's own style banks."""

from diversify_text.method.tinystyler import TinyStylerMethod


class TestTinyStylerBank:

    def test_all_styles_have_example_texts(self):
        for bank in (
            TinyStylerMethod.style_bank,
            TinyStylerMethod.unusual_style_bank,
            TinyStylerMethod.surface_style_bank,
        ):
            for examples in bank.values():
                assert examples
                assert all(isinstance(x, str) for x in examples)
