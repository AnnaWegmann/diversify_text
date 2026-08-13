"""Tests for core Diversifier behaviour: method architecture, batching, convenience function."""

import json
import tempfile
import unittest
from pathlib import Path

from diversify_text import Diversifier, diversify

from tests.fixtures import CountingMethod, FailingMethod, PrefixMethod


class TestDiversifier(unittest.TestCase):

    def test_single_text_returns_one_result(self):
        div = Diversifier(method="echo")
        results = div.diversify("hello")
        self.assertEqual(len(results), 1)
        self.assertIn("original", results[0])
        self.assertIn("paraphrases", results[0])
        self.assertEqual(results[0]["original"], "hello")

    def test_n_controls_paraphrase_count(self):
        div = Diversifier(method="echo")
        results = div.diversify("hello", n=3)
        self.assertEqual(len(results[0]["paraphrases"]), 3)

    def test_multiple_texts(self):
        div = Diversifier(method="echo")
        results = div.diversify(["a", "b", "c"])
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIn("original", r)
            self.assertIn("paraphrases", r)

    def test_custom_method_instance(self):
        div = Diversifier(method=PrefixMethod("x"))
        results = div.diversify("hello", n=3)
        # Each paraphrase is labeled with the default style it belongs to.
        self.assertEqual(
            results[0]["paraphrases"],
            [
                {"style": "informational", "text": "x:hello:0"},
                {"style": "digital_communication", "text": "x:hello:1"},
                {"style": "barackobama", "text": "x:hello:2"},
            ],
        )

    def test_failing_method_raises(self):
        div = Diversifier(method=FailingMethod())
        with self.assertRaises(RuntimeError):
            div.diversify("hello", n=2)

    def test_unknown_method_raises_before_generation(self):
        with self.assertRaises(KeyError):
            Diversifier(method="does_not_exist")

    def test_batching_splits_into_correct_number_of_calls(self):
        method = CountingMethod()
        div = Diversifier(method=method)
        results = div.diversify(["a", "b", "c", "d", "e"], n=2, batch_size=2)
        self.assertEqual(method.calls, 3)
        self.assertEqual(len(results), 5)
        self.assertEqual(
            results[0]["paraphrases"],
            [
                {"style": "informational", "text": "a:0"},
                {"style": "digital_communication", "text": "a:1"},
            ],
        )

    def test_iterator_input_with_output_dir(self):
        def gen():
            yield "one"
            yield "two"

        div = Diversifier(method="echo")
        with tempfile.TemporaryDirectory() as tmpdir:
            result = div.diversify(
                gen(), n=2, output_dir=tmpdir, output_name="out"
            )
            self.assertIsInstance(result, Path)
            out = Path(tmpdir) / "out.jsonl"
            self.assertTrue(out.exists())
            lines = out.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 2)
            record = json.loads(lines[0])
            self.assertEqual(record["original"], "one")
            self.assertEqual(len(record["paraphrases"]), 2)

    def test_iterator_without_output_dir_defaults_to_cwd(self):
        div = Diversifier(method="echo")
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            os.chdir(tmpdir)
            try:
                result = div.diversify(iter(["a", "b"]), n=1)
                self.assertIsInstance(result, Path)
                expected = Path(tmpdir) / "diversified_output.jsonl"
                self.assertTrue(expected.exists())
            finally:
                os.chdir(original_cwd)

    def test_csv_file_writes_jsonl(self):
        div = Diversifier(method="echo")
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            csv_path.write_text("text\nhello\nworld\n", encoding="utf-8")

            result = div.diversify(str(csv_path), text_column="text", n=2)
            self.assertIsInstance(result, Path)

            jsonl_path = Path(tmpdir) / "data_diversified.jsonl"
            self.assertTrue(jsonl_path.exists())
            lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 2)

    def test_txt_file_writes_jsonl(self):
        div = Diversifier(method="echo")
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path = Path(tmpdir) / "texts.txt"
            txt_path.write_text("line one\nline two\n", encoding="utf-8")

            result = div.diversify(str(txt_path), n=2)
            self.assertIsInstance(result, Path)

            jsonl = Path(tmpdir) / "texts.jsonl"
            self.assertTrue(jsonl.exists())
            lines = jsonl.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 2)


class TestDiversifyFunction(unittest.TestCase):

    def test_basic_call(self):
        results = diversify("test input", method="echo")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["original"], "test input")

    def test_list_input(self):
        results = diversify(["a", "b"], n=2, method="echo")
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertEqual(len(r["paraphrases"]), 2)


if __name__ == "__main__":
    unittest.main()
