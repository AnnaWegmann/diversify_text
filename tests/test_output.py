"""Tests for output writing (diversify._output.OutputWriter, resolve_output_path)."""

import json
import tempfile
import unittest
from pathlib import Path

from diversify._input import InputContext, InputKind
from diversify._output import OutputWriter, resolve_output_path


class TestResolveOutputPath(unittest.TestCase):

    # -- In-memory defaults (no output_dir) --

    def test_list_no_output_dir_returns_none(self):
        input_context = InputContext(kind=InputKind.LIST, total=2)
        self.assertIsNone(resolve_output_path(input_context))

    def test_single_str_no_output_dir_returns_none(self):
        input_context = InputContext(kind=InputKind.SINGLE_STR, total=1)
        self.assertIsNone(resolve_output_path(input_context))

    # -- File input defaults (no output_dir, uses input file's directory) --

    def test_csv_default_output_is_jsonl(self):
        input_context = InputContext(
            kind=InputKind.FILE_CSV, input_path=Path("/tmp/data.csv")
        )
        result = resolve_output_path(input_context)
        self.assertEqual(result, Path("/tmp/data_diversified.jsonl"))

    def test_tsv_default_output_is_jsonl(self):
        input_context = InputContext(
            kind=InputKind.FILE_TSV, input_path=Path("/tmp/data.tsv")
        )
        result = resolve_output_path(input_context)
        self.assertEqual(result, Path("/tmp/data_diversified.jsonl"))

    def test_txt_default_output_is_jsonl(self):
        input_context = InputContext(
            kind=InputKind.FILE_TXT, input_path=Path("/tmp/texts.txt")
        )
        result = resolve_output_path(input_context)
        self.assertEqual(result, Path("/tmp/texts.jsonl"))

    # -- Iterable defaults to CWD --

    def test_iterable_no_output_dir_uses_cwd(self):
        input_context = InputContext(kind=InputKind.ITERABLE)
        result = resolve_output_path(input_context)
        self.assertEqual(result, Path.cwd() / "diversified_output.jsonl")

    # -- Explicit output_dir --

    def test_output_dir_overrides_default_directory(self):
        input_context = InputContext(
            kind=InputKind.FILE_CSV, input_path=Path("/tmp/data.csv")
        )
        result = resolve_output_path(input_context, output_dir="/out")
        self.assertEqual(result, Path("/out/data_diversified.jsonl"))

    def test_list_with_output_dir_writes_to_disk(self):
        input_context = InputContext(kind=InputKind.LIST, total=2)
        result = resolve_output_path(input_context, output_dir="/out")
        self.assertEqual(result, Path("/out/diversified_output.jsonl"))

    # -- Explicit output_name --

    def test_output_name_overrides_default_stem(self):
        input_context = InputContext(
            kind=InputKind.FILE_CSV, input_path=Path("/tmp/data.csv")
        )
        result = resolve_output_path(input_context, output_name="my_results")
        self.assertEqual(result, Path("/tmp/my_results.jsonl"))

    # -- Both output_dir and output_name --

    def test_output_dir_and_name_together(self):
        input_context = InputContext(
            kind=InputKind.FILE_CSV, input_path=Path("/tmp/data.csv")
        )
        result = resolve_output_path(input_context, output_dir="/out", output_name="custom")
        self.assertEqual(result, Path("/out/custom.jsonl"))


class TestOutputWriterInMemory(unittest.TestCase):

    def test_accumulates_list_of_dicts(self):
        input_context = InputContext(kind=InputKind.LIST, total=2)
        writer = OutputWriter(input_context, n_styles=2, output_path=None)
        writer.open()
        writer.write_batch(["a", "b"], [["a1", "a2"], ["b1", "b2"]])
        result = writer.finish()
        self.assertEqual(result, [
            {"original": "a", "paraphrases": ["a1", "a2"]},
            {"original": "b", "paraphrases": ["b1", "b2"]},
        ])

    def test_multiple_batches_accumulate(self):
        input_context = InputContext(kind=InputKind.LIST, total=3)
        writer = OutputWriter(input_context, n_styles=1, output_path=None)
        writer.open()
        writer.write_batch(["a"], [["a1"]])
        writer.write_batch(["b", "c"], [["b1"], ["c1"]])
        result = writer.finish()
        self.assertEqual(len(result), 3)


class TestOutputWriterJSONL(unittest.TestCase):

    def test_csv_input_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.jsonl"
            input_context = InputContext(kind=InputKind.FILE_CSV, input_path=Path("x.csv"))
            writer = OutputWriter(input_context, n_styles=2, output_path=out)
            writer.open()
            writer.write_batch(["hello", "world"], [["h1", "h2"], ["w1", "w2"]])
            result = writer.finish()

            self.assertEqual(result, out)
            lines = out.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 2)
            record = json.loads(lines[0])
            self.assertEqual(record["original"], "hello")
            self.assertEqual(record["paraphrases"], ["h1", "h2"])

    def test_iterable_input_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.jsonl"
            input_context = InputContext(kind=InputKind.ITERABLE)
            writer = OutputWriter(input_context, n_styles=1, output_path=out)
            writer.open()
            writer.write_batch(["a"], [["a1"]])
            result = writer.finish()

            self.assertEqual(result, out)
            record = json.loads(out.read_text(encoding="utf-8").strip())
            self.assertEqual(record["original"], "a")


    def test_txt_input_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "texts.jsonl"
            input_context = InputContext(kind=InputKind.FILE_TXT, input_path=Path("texts.txt"))
            writer = OutputWriter(input_context, n_styles=2, output_path=out)
            writer.open()
            writer.write_batch(
                ["one", "two"],
                [["one_s1", "one_s2"], ["two_s1", "two_s2"]],
            )
            result = writer.finish()

            self.assertEqual(result, out)
            lines = out.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 2)
            record = json.loads(lines[0])
            self.assertEqual(record["original"], "one")
            self.assertEqual(record["paraphrases"], ["one_s1", "one_s2"])


if __name__ == "__main__":
    unittest.main()
