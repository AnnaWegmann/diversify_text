"""Tests for resolve_style_dict: input shapes, ordering, naming, and errors."""

import unittest

from diversify_text.styles import resolve_style_dict

# Small toy bank so assertions don't depend on the real bank's content.
# Bank order defines the style indices: 0=recipe, 1=formal, 2=poem.
_BANK = {
    "recipe": ["Cut a peeled brown onion.", "Place one slice on the pastry."],
    "formal": ["He has a very distinct walk."],
    "poem": ["Out there, my old life tempts."],
}


class TestResolveStyleDict(unittest.TestCase):

    def test_select_bank_style_by_name(self):
        result = resolve_style_dict(styles=["recipe"], bank=_BANK)
        self.assertEqual(result, {"recipe": _BANK["recipe"]})

    def test_select_bank_style_by_index(self):
        result = resolve_style_dict(styles=[1], bank=_BANK)
        self.assertEqual(result, {"formal": _BANK["formal"]})

    def test_flat_list_is_one_style_named_style_1(self):
        result = resolve_style_dict(style_examples=["ex a", "ex b"])
        self.assertEqual(result, {"style_1": ["ex a", "ex b"]})

    def test_list_of_lists_gives_one_style_per_list(self):
        result = resolve_style_dict(style_examples=[["a1", "a2"], ["b1"]])
        self.assertEqual(result, {"style_1": ["a1", "a2"], "style_2": ["b1"]})

    def test_dict_keeps_style_names(self):
        result = resolve_style_dict(style_examples={"academic": ["ex 1"]})
        self.assertEqual(result, {"academic": ["ex 1"]})

    def test_combined_call_puts_bank_styles_first(self):
        result = resolve_style_dict(
            styles=["formal"],
            style_examples={"academic": ["ex 1"]},
            bank=_BANK,
        )
        self.assertEqual(
            result,
            {"formal": _BANK["formal"], "academic": ["ex 1"]},
        )
        self.assertEqual(list(result), ["formal", "academic"])

    def test_user_style_clashing_with_bank_style_is_renamed(self):
        result = resolve_style_dict(
            styles=["recipe"],
            style_examples={"recipe": ["my own example"]},
            bank=_BANK,
        )
        self.assertEqual(
            result,
            {"recipe": _BANK["recipe"], "recipe_user": ["my own example"]},
        )

    def test_unknown_style_name_raises(self):
        with self.assertRaises(ValueError) as cm:
            resolve_style_dict(styles=["nonexistent"], bank=_BANK)
        self.assertEqual(
            str(cm.exception),
            "Unknown style 'nonexistent'. "
            "Available: ['formal', 'poem', 'recipe']",
        )

    def test_same_bank_style_twice_raises(self):
        # Index 0 is "recipe", so this requests the same style twice.
        with self.assertRaises(ValueError) as cm:
            resolve_style_dict(styles=[0, "recipe"], bank=_BANK)
        self.assertEqual(
            str(cm.exception),
            "Style 'recipe' requested more than once in styles.",
        )


if __name__ == "__main__":
    unittest.main()
