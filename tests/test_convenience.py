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
        results = diversify("hello", n=7, method=_FakeTinyStyler())
        self.assertEqual(len(results[0]["paraphrases"]), 7)

    def test_n_larger_than_available_styles_raises(self):
        bank_size = len(DEFAULT_STYLE_BANK)
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
            styles=["opinion", "welsh_english"],
        )
        self.assertEqual(len(results[0]["paraphrases"]), 2)

    def test_surface_style_selectable_by_name(self):
        results = diversify("hello", method="echo", styles=["all_caps"])
        self.assertEqual(
            [p['style'] for p in results[0]["paraphrases"]],
            ["all_caps"],
        )

    def test_own_style_examples_give_one_paraphrase_each(self):
        results = diversify(
            "hello",
            method=_FakePrompting(),
            style_texts={"a": ["example a"], "b": ["example b"]},
        )
        self.assertEqual(len(results[0]["paraphrases"]), 2)

    def test_n_combined_with_styles_raises(self):
        with self.assertRaises(ValueError) as cm:
            diversify(
                "hello",
                n=3,
                method=_FakeTinyStyler(),
                styles=["scottish_english"],
            )
        self.assertEqual(
            str(cm.exception),
            "n cannot be combined with styles or style_texts — the "
            "number of styles already determines the number of paraphrases.",
        )

    def test_repeats_interleave_styles(self):
        results = diversify(
            "hello",
            method=_FakeTinyStyler(),
            style_texts={"a": ["example a"], "b": ["example b"]},
            repeats=2,
        )
        # Two styles, two repeats → four paraphrases: a, b, a, b.
        self.assertEqual(
            [p["style"] for p in results[0]["paraphrases"]],
            ["a", "b", "a", "b"],
        )

    def test_repeats_below_one_raises(self):
        with self.assertRaises(ValueError) as cm:
            diversify("hello", method=_FakeTinyStyler(), repeats=0)
        self.assertEqual(str(cm.exception), "repeats must be >= 1.")

    def test_method_style_bank_drives_selection(self):
        """styles and the n-default pool come from the active method's bank."""

        class _TwoStyleMethod(PrefixMethod):
            name = "two_style"
            style_bank = {"a": ["ex a"], "b": ["ex b"]}

        # Plain call: the default n is capped at the bank size.
        results = diversify("hello", method=_TwoStyleMethod("ts"))
        self.assertEqual(
            [p["style"] for p in results[0]["paraphrases"]],
            ["a", "b"],
        )

        # Name selection resolves against the method's own bank.
        results = diversify("hello", method=_TwoStyleMethod("ts"), styles=["b"])
        self.assertEqual(
            [p["style"] for p in results[0]["paraphrases"]],
            ["b"],
        )

    def test_prompt_selection_does_not_affect_count(self):
        results = diversify(
            "hello",
            method=_FakePrompting(),
            method_kwargs={"prompt": "humanize_transfer"},
        )
        self.assertEqual(len(results[0]["paraphrases"]), _DEFAULT_N)


class TestModelSelection(unittest.TestCase):
    """The model keyword configures methods that take a model choice."""

    def test_model_with_tinystyler_raises(self):
        with self.assertRaises(ValueError) as cm:
            Diversifier(method="tinystyler", model="my/model")
        self.assertEqual(
            str(cm.exception),
            "The 'tinystyler' method does not accept a model choice "
            "(it uses a fixed model).",
        )

    def test_model_given_twice_raises(self):
        with self.assertRaises(ValueError) as cm:
            diversify(
                "hello",
                method="prompting",
                model="my/model",
                method_kwargs={"model": "other/model"},
            )
        self.assertEqual(
            str(cm.exception),
            "Pass the model either as model=... or inside "
            "method_kwargs, not both.",
        )


if __name__ == "__main__":
    unittest.main()
