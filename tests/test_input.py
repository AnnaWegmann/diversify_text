"""Tests for input normalisation (diversify._io.normalize_input)."""

import unittest

import pandas as pd

from diversify._io import normalize_input


class TestNormalizeInput(unittest.TestCase):

    def test_single_string(self):
        self.assertEqual(normalize_input("hello", text_column="text"), ["hello"])

    def test_list_of_strings(self):
        self.assertEqual(normalize_input(["a", "b", "c"], text_column="text"), ["a", "b", "c"])

    def test_pandas_series(self):
        self.assertEqual(normalize_input(pd.Series(["x", "y"]), text_column="text"), ["x", "y"])

    def test_pandas_dataframe(self):
        df = pd.DataFrame({"text": ["one", "two"], "other": [1, 2]})
        self.assertEqual(normalize_input(df, text_column="text"), ["one", "two"])

    def test_dataframe_custom_column(self):
        df = pd.DataFrame({"bio": ["a", "b"]})
        self.assertEqual(normalize_input(df, text_column="bio"), ["a", "b"])

    def test_unsupported_type_raises(self):
        with self.assertRaises(TypeError):
            normalize_input(42, text_column="text")


if __name__ == "__main__":
    unittest.main()
