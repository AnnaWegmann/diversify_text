"""Tests for tabular I/O: CSV, TSV, DataFrame input/output, and punctuation splitting."""

import unittest
from pathlib import Path
import tempfile

import pandas as pd

from diversify import Diversifier


class TestTabularIO(unittest.TestCase):

    def setUp(self):
        self.div = Diversifier(methods=["echo"])

    def test_dataframe_input_returns_dataframe_with_style_columns(self):
        df = pd.DataFrame({"text": ["one", "two"]})
        results = self.div.diversify(df, text_column="text")
        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(len(results), 2)
        self.assertIn("style 1", results.columns)
        self.assertIn("style 5", results.columns)
        self.assertEqual(results.loc[0, "style 1"], "one")
        self.assertEqual(results.loc[1, "style 1"], "two")

    def test_csv_input_returns_dataframe_and_saves_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "bios.csv"
            pd.DataFrame({"bio": ["one", "two"], "id": [1, 2]}).to_csv(input_path, index=False)

            results = self.div.diversify(str(input_path), text_column="bio", n_styles=2)
            self.assertIsInstance(results, pd.DataFrame)
            self.assertIn("style 1", results.columns)
            self.assertIn("style 2", results.columns)

            output_path = Path(tmpdir) / "bios_diversified.csv"
            self.assertTrue(output_path.exists())
            self.assertIn("style 1", pd.read_csv(output_path).columns)

    def test_tsv_input_saves_tsv_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "bios.tsv"
            pd.DataFrame({"bio": ["one", "two"], "id": [1, 2]}).to_csv(
                input_path, sep="\t", index=False
            )
            self.div.diversify(str(input_path), text_column="bio", n_styles=1)

            output_path = Path(tmpdir) / "bios_diversified.tsv"
            self.assertTrue(output_path.exists())
            self.assertIn("style 1", pd.read_csv(output_path, sep="\t").columns)


if __name__ == "__main__":
    unittest.main()
