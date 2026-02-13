"""Tests for the core diversify API."""

import unittest

import pandas as pd

from diversify import Diversifier, diversify
from diversify.core import DiversificationMethod


class PrefixMethod(DiversificationMethod):
    """Simple fake method for unit testing."""

    name = "prefix"

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def generate(
        self,
        texts,
        *,
        n_styles,
        max_new_tokens,
        temperature,
        top_p,
        **kwargs,
    ):
        return [
            [f"{self.prefix}:{text}:{i}" for i in range(n_styles)]
            for text in texts
        ]


class FailingMethod(DiversificationMethod):
    name = "failing"

    def generate(
        self,
        texts,
        *,
        n_styles,
        max_new_tokens,
        temperature,
        top_p,
        **kwargs,
    ):
        raise RuntimeError("boom")


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
        self.div = Diversifier(methods=["echo"])

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
        results = diversify("test input", methods=["echo"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["original"], "test input")

    def test_list_input(self):
        results = diversify(["a", "b"], n_styles=2, methods=["echo"])
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertEqual(len(r["paraphrases"]), 2)


class TestMethodArchitecture(unittest.TestCase):
    def test_custom_method_instance(self):
        div = Diversifier(methods=[PrefixMethod("x")])
        results = div.diversify("hello", n_styles=3)
        self.assertEqual(results[0]["paraphrases"], ["x:hello:0", "x:hello:1", "x:hello:2"])

    def test_multiple_methods_distribute_styles(self):
        div = Diversifier(methods=[PrefixMethod("a"), PrefixMethod("b")])
        results = div.diversify("hello", n_styles=5)
        paraphrases = results[0]["paraphrases"]
        self.assertEqual(len(paraphrases), 5)
        self.assertEqual(paraphrases[:3], ["a:hello:0", "a:hello:1", "a:hello:2"])
        self.assertEqual(paraphrases[3:], ["b:hello:0", "b:hello:1"])

    def test_fallback_warns_when_method_fails(self):
        div = Diversifier(methods=[FailingMethod()], fallback_method="echo")
        with self.assertWarns(RuntimeWarning):
            results = div.diversify("hello", n_styles=2)
        self.assertEqual(results[0]["paraphrases"], ["hello", "hello"])


if __name__ == "__main__":
    unittest.main()
