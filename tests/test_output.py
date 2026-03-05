"""Tests for output writing (diversify._io.OutputWriter, resolve_output_path)."""

import json
import tempfile
import unittest
from pathlib import Path

from diversify._io import (
    InputContext,
    InputKind,
    OutputWriter,
    resolve_output_path,
)


class TestResolveOutputPath(unittest.TestCase):

    def test_list_no_output_path_returns_none(self):
        ctx = InputContext(kind=InputKind.LIST, total=2)
        self.assertIsNone(resolve_output_path(ctx, output_path=None))

    def test_single_str_no_output_path_returns_none(self):
        ctx = InputContext(kind=InputKind.SINGLE_STR, total=1)
        self.assertIsNone(resolve_output_path(ctx, output_path=None))

    def test_iterable_without_output_path_raises(self):
        ctx = InputContext(kind=InputKind.ITERABLE)
        with self.assertRaises(ValueError):
            resolve_output_path(ctx, output_path=None)

    def test_csv_default_output_is_jsonl(self):
        ctx = InputContext(
            kind=InputKind.FILE_CSV, input_path=Path("/tmp/data.csv")
        )
        result = resolve_output_path(ctx, output_path=None)
        self.assertEqual(result, Path("/tmp/data_diversified.jsonl"))

    def test_tsv_default_output_is_jsonl(self):
        ctx = InputContext(
            kind=InputKind.FILE_TSV, input_path=Path("/tmp/data.tsv")
        )
        result = resolve_output_path(ctx, output_path=None)
        self.assertEqual(result, Path("/tmp/data_diversified.jsonl"))

    def test_txt_default_output_is_input_path(self):
        ctx = InputContext(
            kind=InputKind.FILE_TXT, input_path=Path("/tmp/texts.txt")
        )
        result = resolve_output_path(ctx, output_path=None)
        self.assertEqual(result, Path("/tmp/texts.txt"))

    def test_explicit_output_path_overrides_default(self):
        ctx = InputContext(
            kind=InputKind.FILE_CSV, input_path=Path("/tmp/data.csv")
        )
        result = resolve_output_path(ctx, output_path="/out/custom.jsonl")
        self.assertEqual(result, Path("/out/custom.jsonl"))


class TestOutputWriterInMemory(unittest.TestCase):

    def test_accumulates_list_of_dicts(self):
        ctx = InputContext(kind=InputKind.LIST, total=2)
        writer = OutputWriter(ctx, n_styles=2, output_path=None)
        writer.open()
        writer.write_batch(["a", "b"], [["a1", "a2"], ["b1", "b2"]])
        result = writer.finish()
        self.assertEqual(result, [
            {"original": "a", "paraphrases": ["a1", "a2"]},
            {"original": "b", "paraphrases": ["b1", "b2"]},
        ])

    def test_multiple_batches_accumulate(self):
        ctx = InputContext(kind=InputKind.LIST, total=3)
        writer = OutputWriter(ctx, n_styles=1, output_path=None)
        writer.open()
        writer.write_batch(["a"], [["a1"]])
        writer.write_batch(["b", "c"], [["b1"], ["c1"]])
        result = writer.finish()
        self.assertEqual(len(result), 3)


class TestOutputWriterJSONL(unittest.TestCase):

    def test_csv_input_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.jsonl"
            ctx = InputContext(kind=InputKind.FILE_CSV, input_path=Path("x.csv"))
            writer = OutputWriter(ctx, n_styles=2, output_path=out)
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
            ctx = InputContext(kind=InputKind.ITERABLE)
            writer = OutputWriter(ctx, n_styles=1, output_path=out)
            writer.open()
            writer.write_batch(["a"], [["a1"]])
            result = writer.finish()

            self.assertEqual(result, out)
            record = json.loads(out.read_text(encoding="utf-8").strip())
            self.assertEqual(record["original"], "a")


class TestOutputWriterTxt(unittest.TestCase):

    def test_txt_input_creates_n_style_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "texts.txt"
            ctx = InputContext(kind=InputKind.FILE_TXT, input_path=base)
            writer = OutputWriter(ctx, n_styles=2, output_path=base)
            writer.open()
            writer.write_batch(
                ["one", "two"],
                [["one_s1", "one_s2"], ["two_s1", "two_s2"]],
            )
            result = writer.finish()

            self.assertEqual(result, base)
            f1 = base.with_name("texts_diversified_1.txt")
            f2 = base.with_name("texts_diversified_2.txt")
            self.assertTrue(f1.exists())
            self.assertTrue(f2.exists())
            self.assertEqual(
                f1.read_text(encoding="utf-8").strip().split("\n"),
                ["one_s1", "two_s1"],
            )
            self.assertEqual(
                f2.read_text(encoding="utf-8").strip().split("\n"),
                ["one_s2", "two_s2"],
            )


if __name__ == "__main__":
    unittest.main()
