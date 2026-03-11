"""Tests for input resolution (diversify._io.resolve_input)."""

import tempfile
import unittest
from pathlib import Path

from diversify_text._input import InputKind, resolve_input


class TestResolveInput(unittest.TestCase):

    def test_single_string_not_file(self):
        text_iter, input_context = resolve_input("hello world")
        self.assertEqual(list(text_iter), ["hello world"])
        self.assertEqual(input_context.kind, InputKind.SINGLE_STR)
        self.assertEqual(input_context.total, 1)

    def test_list_of_strings(self):
        text_iter, input_context = resolve_input(["a", "b", "c"])
        self.assertEqual(list(text_iter), ["a", "b", "c"])
        self.assertEqual(input_context.kind, InputKind.LIST)
        self.assertEqual(input_context.total, 3)

    def test_generator_input(self):
        def gen():
            yield "x"
            yield "y"

        text_iter, input_context = resolve_input(gen())
        self.assertEqual(input_context.kind, InputKind.ITERABLE)
        self.assertIsNone(input_context.total)  # should be None for iterables of unknown length
        self.assertEqual(list(text_iter), ["x", "y"])

    def test_csv_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "data.csv"
            p.write_text("id,text\n1,hello\n2,world\n", encoding="utf-8")
            text_iter, input_context = resolve_input(str(p), text_column="text")
            self.assertEqual(input_context.kind, InputKind.FILE_CSV)
            self.assertEqual(input_context.total, 2)
            self.assertEqual(input_context.input_path, p)
            self.assertEqual(list(text_iter), ["hello", "world"])

    def test_tsv_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "data.tsv"
            p.write_text("id\ttext\n1\thello\n2\tworld\n", encoding="utf-8")
            text_iter, input_context = resolve_input(str(p), text_column="text")
            self.assertEqual(input_context.kind, InputKind.FILE_TSV)
            self.assertEqual(input_context.total, 2)
            self.assertEqual(list(text_iter), ["hello", "world"])

    def test_txt_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "texts.txt"
            p.write_text("line one\nline two\nline three\n", encoding="utf-8")
            text_iter, input_context = resolve_input(str(p))
            self.assertEqual(input_context.kind, InputKind.FILE_TXT)
            self.assertEqual(input_context.total, 3)
            self.assertEqual(list(text_iter), ["line one", "line two", "line three"])

    def test_unsupported_type_raises(self):
        with self.assertRaises(TypeError):
            resolve_input(42)


class TestResolveInputEdgeCases(unittest.TestCase):

    def test_txt_file_skips_empty_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "texts.txt"
            p.write_text("one\n\ntwo\n  \nthree\n", encoding="utf-8")
            text_iter, input_context = resolve_input(str(p))
            self.assertEqual(input_context.total, 3)
            self.assertEqual(list(text_iter), ["one", "two", "three"])

    def test_csv_missing_column_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "data.csv"
            p.write_text("id,name\n1,alice\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                resolve_input(str(p), text_column="text")


if __name__ == "__main__":
    unittest.main()
