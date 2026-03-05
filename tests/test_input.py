"""Tests for input resolution (diversify._io.resolve_input)."""

import tempfile
import unittest
from pathlib import Path

from diversify._io import InputKind, resolve_input


class TestResolveInput(unittest.TestCase):

    def test_single_string_not_file(self):
        it, ctx = resolve_input("hello world")
        self.assertEqual(list(it), ["hello world"])
        self.assertEqual(ctx.kind, InputKind.SINGLE_STR)
        self.assertEqual(ctx.total, 1)

    def test_list_of_strings(self):
        it, ctx = resolve_input(["a", "b", "c"])
        self.assertEqual(list(it), ["a", "b", "c"])
        self.assertEqual(ctx.kind, InputKind.LIST)
        self.assertEqual(ctx.total, 3)

    def test_generator_input(self):
        def gen():
            yield "x"
            yield "y"

        it, ctx = resolve_input(gen())
        self.assertEqual(ctx.kind, InputKind.ITERABLE)
        self.assertIsNone(ctx.total)
        self.assertEqual(list(it), ["x", "y"])

    def test_csv_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "data.csv"
            p.write_text("id,text\n1,hello\n2,world\n", encoding="utf-8")
            it, ctx = resolve_input(str(p), text_column="text")
            self.assertEqual(ctx.kind, InputKind.FILE_CSV)
            self.assertEqual(ctx.total, 2)
            self.assertEqual(ctx.input_path, p)
            self.assertEqual(list(it), ["hello", "world"])

    def test_tsv_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "data.tsv"
            p.write_text("id\ttext\n1\thello\n2\tworld\n", encoding="utf-8")
            it, ctx = resolve_input(str(p), text_column="text")
            self.assertEqual(ctx.kind, InputKind.FILE_TSV)
            self.assertEqual(ctx.total, 2)
            self.assertEqual(list(it), ["hello", "world"])

    def test_txt_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "texts.txt"
            p.write_text("line one\nline two\nline three\n", encoding="utf-8")
            it, ctx = resolve_input(str(p))
            self.assertEqual(ctx.kind, InputKind.FILE_TXT)
            self.assertEqual(ctx.total, 3)
            self.assertEqual(list(it), ["line one", "line two", "line three"])

    def test_txt_file_skips_empty_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "texts.txt"
            p.write_text("one\n\ntwo\n  \nthree\n", encoding="utf-8")
            it, ctx = resolve_input(str(p))
            self.assertEqual(ctx.total, 3)
            self.assertEqual(list(it), ["one", "two", "three"])

    def test_csv_missing_column_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "data.csv"
            p.write_text("id,name\n1,alice\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                resolve_input(str(p), text_column="text")

    def test_csv_with_empty_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "data.csv"
            # CSV with an explicitly empty cell (not a blank line)
            p.write_text("id,text\n1,hello\n2,\n3,world\n", encoding="utf-8")
            it, ctx = resolve_input(str(p), text_column="text")
            self.assertEqual(list(it), ["hello", "", "world"])

    def test_string_with_csv_extension_but_no_file(self):
        it, ctx = resolve_input("nonexistent.csv")
        self.assertEqual(ctx.kind, InputKind.SINGLE_STR)
        self.assertEqual(list(it), ["nonexistent.csv"])

    def test_unsupported_type_raises(self):
        with self.assertRaises(TypeError):
            resolve_input(42)


if __name__ == "__main__":
    unittest.main()
