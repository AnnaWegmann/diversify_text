"""Tests for the convenience ``diversify()`` function: n inference from method_kwargs."""

import unittest

from diversify_text import Diversifier, diversify

from tests.fixtures import PrefixMethod

_DEFAULT_N = Diversifier._DEFAULT_N


# -- Add Fixtures ----------------------------------------------------------

# PrefixMethod is a simple fixture that returns exactly n "paraphrases" per text,
# so we subclass it and only override ``name`` to trigger the right inference branch
# without loading real models.


class _FakeTinyStyler(PrefixMethod):
    name = "tinystyler"

    def __init__(self) -> None:
        super().__init__("ts")


class _FakePrompting(PrefixMethod):
    name = "prompting"

    def __init__(self) -> None:
        super().__init__("pr")


class TestInferNFromMethodKwargs(unittest.TestCase):
    """Verify that n is correctly inferred from method_kwargs when omitted."""

    # -- tinystyler ----------------------------------------------------------

    def test_tinystyler_n_inferred_from_styles(self):
        """n=None + 3 styles → 3 paraphrases."""
        results = diversify(
            "hello",
            method=_FakeTinyStyler(),
            method_kwargs={"styles": ["a", "b", "c"]},
        )
        self.assertEqual(len(results[0]["paraphrases"]), 3)

    def test_tinystyler_explicit_n_overrides_styles_len(self):
        """n=10 + 1 style → 10 paraphrases (n wins)."""
        results = diversify(
            "hello",
            n=10,
            method=_FakeTinyStyler(),
            method_kwargs={"styles": ["informal_tinystyler"]},
        )
        self.assertEqual(len(results[0]["paraphrases"]), 10)

    def test_tinystyler_no_kwargs_defaults_to_default_n(self):
        """n=None + no method_kwargs → _DEFAULT_N."""
        results = diversify("hello", method=_FakeTinyStyler())
        self.assertEqual(len(results[0]["paraphrases"]), _DEFAULT_N)

    def test_tinystyler_n_inferred_from_style_bank(self):
        """n=None + a 2-style custom bank (no styles selection) → 2 paraphrases."""
        results = diversify(
            "hello",
            method=_FakeTinyStyler(),
            method_kwargs={"style_bank": {
                "a": ["example a"], "b": ["example b"],
            }},
        )
        self.assertEqual(len(results[0]["paraphrases"]), 2)

    # -- prompting: prompt selection ------------------------------------------

    def test_prompting_prompt_selection_does_not_affect_n(self):
        """n=None + a prompt selection (no styles) → _DEFAULT_N."""
        results = diversify(
            "hello",
            method=_FakePrompting(),
            method_kwargs={"prompt": "humanize_transfer"},
        )
        self.assertEqual(len(results[0]["paraphrases"]), _DEFAULT_N)

    def test_prompting_n_inferred_from_custom_style_bank(self):
        """n=None + a 2-style custom bank (no styles selection) → 2 paraphrases."""
        results = diversify(
            "hello",
            method=_FakePrompting(),
            method_kwargs={"custom_style_bank": {
                "a": ["example a"], "b": ["example b"],
            }},
        )
        self.assertEqual(len(results[0]["paraphrases"]), 2)

    def test_prompting_no_kwargs_defaults_to_default_n(self):
        """n=None + no method_kwargs → _DEFAULT_N."""
        results = diversify("hello", method=_FakePrompting())
        self.assertEqual(len(results[0]["paraphrases"]), _DEFAULT_N)

    # -- prompting: styles only ----------------------------------------------

    def test_prompting_n_inferred_from_styles(self):
        """n=None + 3 styles → 3 paraphrases."""
        results = diversify(
            "hello",
            method=_FakePrompting(),
            method_kwargs={"styles": ["a", "b", "c"]},
        )
        self.assertEqual(len(results[0]["paraphrases"]), 3)

    def test_prompting_explicit_n_overrides_styles_len(self):
        """n=10 + 1 style → 10 paraphrases (n wins)."""
        results = diversify(
            "hello",
            n=10,
            method=_FakePrompting(),
            method_kwargs={"styles": ["informal_tinystyler"]},
        )
        self.assertEqual(len(results[0]["paraphrases"]), 10)

    # -- prompting: prompt selection + styles --------------------------------

    def test_prompting_prompt_with_styles_infers_from_styles(self):
        """A prompt selection + 3 styles → 3 paraphrases (one per style)."""
        results = diversify(
            "hello",
            method=_FakePrompting(),
            method_kwargs={
                "prompt": "style_transfer",
                "styles": ["a", "b", "c"],
            },
        )
        self.assertEqual(len(results[0]["paraphrases"]), 3)


if __name__ == "__main__":
    unittest.main()
