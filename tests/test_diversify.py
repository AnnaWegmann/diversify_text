"""Tests for the core diversify API."""

import unittest

import pandas as pd

from diversify import Diversifier, diversify


class TestNormalizeInput(unittest.TestCase):
    """Verify that Diversifier._normalize_input handles all input types."""

    def setUp(self):
        self.div = Diversifier()

    def test_single_string(self):
        result = self.div._normalize_input("hello", text_column="text")
        self.assertEqual(result, ["hello"])

    def test_list_of_strings(self):
        texts = ["a", "b", "c"]
        result = self.div._normalize_input(texts, text_column="text")
        self.assertEqual(result, ["a", "b", "c"])

    def test_pandas_series(self):
        series = pd.Series(["x", "y"])
        result = self.div._normalize_input(series, text_column="text")
        self.assertEqual(result, ["x", "y"])

    def test_pandas_dataframe(self):
        df = pd.DataFrame({"text": ["one", "two"], "other": [1, 2]})
        result = self.div._normalize_input(df, text_column="text")
        self.assertEqual(result, ["one", "two"])

    def test_dataframe_custom_column(self):
        df = pd.DataFrame({"bio": ["a", "b"]})
        result = self.div._normalize_input(df, text_column="bio")
        self.assertEqual(result, ["a", "b"])

    def test_unsupported_type_raises(self):
        with self.assertRaises(TypeError):
            self.div._normalize_input(42, text_column="text")


class TestDiversifyOutput(unittest.TestCase):
    """Verify the shape / structure of diversify() output."""

    def setUp(self):
        self.div = Diversifier()

    def test_single_text_returns_one_result(self):
        results = self.div.diversify("hello")
        self.assertEqual(len(results), 1)
        self.assertIn("original", results[0])
        self.assertIn("paraphrases", results[0])
        self.assertEqual(results[0]["original"], "hello")

    def test_n_styles_controls_paraphrase_count(self):
        results = self.div.diversify("hello", n_styles=3)
        self.assertEqual(len(results[0]["paraphrases"]), 3)

    def test_multiple_texts(self):
        results = self.div.diversify(["a", "b", "c"])
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIn("original", r)
            self.assertIn("paraphrases", r)

    def test_dataframe_input(self):
        df = pd.DataFrame({"text": ["one", "two"]})
        results = self.div.diversify(df, text_column="text")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["original"], "one")
        self.assertEqual(results[1]["original"], "two")


class TestConvenienceFunction(unittest.TestCase):
    """Verify the module-level diversify() convenience function."""

    def test_basic_call(self):
        results = diversify("test input")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["original"], "test input")

    def test_list_input(self):
        results = diversify(["a", "b"], n_styles=2)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertEqual(len(r["paraphrases"]), 2)


if __name__ == "__main__":
    unittest.main()
