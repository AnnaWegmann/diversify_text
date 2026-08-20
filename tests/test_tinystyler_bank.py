"""Tests for TinyStyler's own style banks."""

from diversify_text.method.tinystyler import TinyStylerMethod


class TestTinyStylerBank:
    def test_default_n_prefix(self):
        # The first five styles are what every default caller (n=5) gets.
        assert list(TinyStylerMethod.style_bank)[:5] == [
            "informal",
            "formal",
            "question",
            "question_answer_forum",
            "discussion_forum",
        ]

    def test_banks_are_disjoint(self):
        assert not (
            set(TinyStylerMethod.style_bank)
            & set(TinyStylerMethod.unusual_style_bank)
        )

    def test_all_styles_have_example_texts(self):
        for bank in (
            TinyStylerMethod.style_bank,
            TinyStylerMethod.unusual_style_bank,
            TinyStylerMethod.surface_style_bank,
        ):
            for examples in bank.values():
                assert examples
                assert all(isinstance(x, str) for x in examples)
