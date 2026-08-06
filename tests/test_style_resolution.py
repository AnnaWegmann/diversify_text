"""Tests for resolve_style_dict: shapes, ordering, auto-naming, and errors."""

import unittest

from diversify_text.styles import DEFAULT_STYLE_BANK, resolve_style_dict

# Small toy bank so assertions don't depend on the real bank's content.
_BANK = {
    "recipe": ["Cut a peeled brown onion.", "Place one slice on the pastry."],
    "formal": ["He has a very distinct walk."],
    "poem": ["Out there, my old life tempts."],
}


class TestBankSelection(unittest.TestCase):

    def test_select_by_name(self):
        result = resolve_style_dict(styles=["recipe"], bank=_BANK)
        self.assertEqual(result, {"recipe": _BANK["recipe"]})

    def test_select_by_index(self):
        result = resolve_style_dict(styles=[1], bank=_BANK)
        self.assertEqual(result, {"formal": _BANK["formal"]})

    def test_mixed_names_and_indices_preserve_order(self):
        result = resolve_style_dict(styles=[2, "recipe"], bank=_BANK)
        self.assertEqual(list(result), ["poem", "recipe"])

    def test_default_bank_used_when_bank_omitted(self):
        result = resolve_style_dict(styles=[0])
        first_name = next(iter(DEFAULT_STYLE_BANK))
        self.assertEqual(list(result), [first_name])

    def test_unknown_name_raises_listing_available(self):
        with self.assertRaises(ValueError) as cm:
            resolve_style_dict(styles=["nonexistent"], bank=_BANK)
        self.assertIn("nonexistent", str(cm.exception))
        self.assertIn("recipe", str(cm.exception))

    def test_out_of_range_index_raises_with_range(self):
        with self.assertRaises(ValueError) as cm:
            resolve_style_dict(styles=[3], bank=_BANK)
        self.assertIn("3 styles", str(cm.exception))

    def test_same_style_by_name_twice_raises(self):
        with self.assertRaises(ValueError) as cm:
            resolve_style_dict(styles=["recipe", "recipe"], bank=_BANK)
        self.assertIn("recipe", str(cm.exception))

    def test_same_style_by_name_and_index_raises(self):
        with self.assertRaises(ValueError):
            resolve_style_dict(styles=[0, "recipe"], bank=_BANK)

    def test_bool_entry_raises_type_error(self):
        with self.assertRaises(TypeError):
            resolve_style_dict(styles=[True], bank=_BANK)


class TestStyleExamples(unittest.TestCase):

    def test_flat_list_is_one_style_named_style_1(self):
        result = resolve_style_dict(style_examples=["ex a", "ex b"])
        self.assertEqual(result, {"style_1": ["ex a", "ex b"]})

    def test_list_of_lists_auto_named_one_based(self):
        result = resolve_style_dict(style_examples=[["a1", "a2"], ["b1"]])
        self.assertEqual(result, {"style_1": ["a1", "a2"], "style_2": ["b1"]})

    def test_dict_keeps_names(self):
        result = resolve_style_dict(style_examples={"academic": ["ex 1"]})
        self.assertEqual(result, {"academic": ["ex 1"]})

    def test_empty_example_set_raises(self):
        with self.assertRaises(ValueError) as cm:
            resolve_style_dict(style_examples={"academic": []})
        self.assertIn("academic", str(cm.exception))

    def test_mixed_list_shapes_raise_type_error(self):
        with self.assertRaises(TypeError):
            resolve_style_dict(style_examples=["a string", ["a", "list"]])

    def test_non_list_dict_value_raises_type_error(self):
        with self.assertRaises(TypeError):
            resolve_style_dict(style_examples={"academic": "not a list"})


class TestCombining(unittest.TestCase):

    def test_bank_styles_come_first(self):
        result = resolve_style_dict(
            styles=["recipe"],
            style_examples={"academic": ["ex 1"]},
            bank=_BANK,
        )
        self.assertEqual(list(result), ["recipe", "academic"])

    def test_clash_with_bank_style_renamed_with_user_suffix(self):
        result = resolve_style_dict(
            styles=["recipe"],
            style_examples={"recipe": ["my own example"]},
            bank=_BANK,
        )
        self.assertEqual(list(result), ["recipe", "recipe_user"])
        self.assertEqual(result["recipe"], _BANK["recipe"])
        self.assertEqual(result["recipe_user"], ["my own example"])

    def test_unresolvable_clash_raises(self):
        # "recipe" must be renamed to "recipe_user", but that name is
        # already taken by a selected bank style → error.
        with self.assertRaises(ValueError) as cm:
            resolve_style_dict(
                styles=["recipe", "recipe_user"],
                style_examples={"recipe": ["a"]},
                bank={**_BANK, "recipe_user": ["bank entry"]},
            )
        self.assertIn("recipe_user", str(cm.exception))

    def test_neither_parameter_raises(self):
        with self.assertRaises(ValueError):
            resolve_style_dict()


if __name__ == "__main__":
    unittest.main()
