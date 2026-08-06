"""Tests for the convenience ``diversify()`` function: style selection and n."""

import unittest

from diversify_text import Diversifier, diversify
from diversify_text.styles import DEFAULT_STYLE_BANK

from tests.fixtures import PrefixMethod

_DEFAULT_N = Diversifier._DEFAULT_N


# -- Add Fixtures ----------------------------------------------------------

# PrefixMethod is a simple fixture that returns one "paraphrase" per style,
# so we subclass it and only override ``name`` to test without loading
# real models.


class _FakeTinyStyler(PrefixMethod):
    name = "tinystyler"

    def __init__(self) -> None:
        super().__init__("ts")


class _FakePrompting(PrefixMethod):
    name = "prompting"

    def __init__(self) -> None:
        super().__init__("pr")


class TestStyleSelection(unittest.TestCase):
    """The number of styles determines the number of paraphrases."""

    def test_default_gives_default_n_paraphrases(self):
        results = diversify("hello", method=_FakeTinyStyler())
        self.assertEqual(len(results[0]["paraphrases"]), _DEFAULT_N)

    def test_n_selects_that_many_bank_styles(self):
        # n draws from the whole style bank, so values beyond the old
        # 5-style default list work.
        results = diversify("hello", n=7, method=_FakeTinyStyler())
        self.assertEqual(len(results[0]["paraphrases"]), 7)

    def test_n_larger_than_available_styles_raises(self):
        bank_size = len(DEFAULT_STYLE_BANK)  # 49 at the time of writing
        with self.assertRaises(ValueError) as cm:
            diversify("hello", n=bank_size + 1, method=_FakeTinyStyler())
        self.assertEqual(
            str(cm.exception),
            f"n={bank_size + 1} exceeds the number of available styles "
            f"({bank_size}).",
        )

    def test_styles_from_bank_give_one_paraphrase_each(self):
        results = diversify(
            "hello",
            method=_FakeTinyStyler(),
            styles=["recipe", "poem"],
        )
        self.assertEqual(len(results[0]["paraphrases"]), 2)

    def test_own_style_examples_give_one_paraphrase_each(self):
        results = diversify(
            "hello",
            method=_FakePrompting(),
            style_examples={"a": ["example a"], "b": ["example b"]},
        )
        self.assertEqual(len(results[0]["paraphrases"]), 2)

    def test_n_combined_with_styles_raises(self):
        with self.assertRaises(ValueError) as cm:
            diversify(
                "hello",
                n=3,
                method=_FakeTinyStyler(),
                styles=["recipe"],
            )
        self.assertEqual(
            str(cm.exception),
            "n cannot be combined with styles or style_examples — the "
            "number of styles already determines the number of paraphrases.",
        )

    def test_prompt_selection_does_not_affect_count(self):
        results = diversify(
            "hello",
            method=_FakePrompting(),
            method_kwargs={"prompt": "humanize_transfer"},
        )
        self.assertEqual(len(results[0]["paraphrases"]), _DEFAULT_N)


if __name__ == "__main__":
    unittest.main()
