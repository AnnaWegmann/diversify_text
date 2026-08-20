"""Tests for TinyStyler's own style banks."""

from diversify_text.method.tinystyler import TinyStylerMethod
from diversify_text.method.tinystyler.bank import (
    TINYSTYLER_STYLE_BANK,
    TINYSTYLER_SURFACE_STYLE_BANK,
    TINYSTYLER_UNUSUAL_STYLE_BANK,
)


class TestTinyStylerBank:
    def test_method_uses_its_own_banks(self):
        assert TinyStylerMethod.style_bank is TINYSTYLER_STYLE_BANK
        assert TinyStylerMethod.unusual_style_bank is TINYSTYLER_UNUSUAL_STYLE_BANK
        assert TinyStylerMethod.surface_style_bank is TINYSTYLER_SURFACE_STYLE_BANK

    def test_default_n_prefix(self):
        # The first five styles are what every default caller (n=5) gets.
        assert list(TINYSTYLER_STYLE_BANK)[:5] == [
            "informal",
            "formal",
            "question",
            "question_answer_forum",
            "discussion_forum",
        ]

    def test_banks_are_disjoint(self):
        assert not set(TINYSTYLER_STYLE_BANK) & set(TINYSTYLER_UNUSUAL_STYLE_BANK)

    def test_all_styles_have_example_texts(self):
        for bank in (
            TINYSTYLER_STYLE_BANK,
            TINYSTYLER_UNUSUAL_STYLE_BANK,
            TINYSTYLER_SURFACE_STYLE_BANK,
        ):
            for examples in bank.values():
                assert examples
                assert all(isinstance(x, str) for x in examples)
