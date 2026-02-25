"""Tests for punctuation splitting — the split_text_on_punctuation function
and the split_on_punctuation=True behaviour in Diversifier."""

import unittest
from pathlib import Path
import tempfile

import pandas as pd

from diversify import Diversifier
from diversify._text import split_text_on_punctuation
from tests.fixtures import PrefixMethod


class TestPunctuationSplitting(unittest.TestCase):

    # --- split_text_on_punctuation ---

    def test_splits_on_period(self):
        self.assertEqual(
            split_text_on_punctuation("First. Second."),
            ["First.", "Second."],
        )

    def test_splits_on_multiple_punctuation_types(self):
        self.assertEqual(
            split_text_on_punctuation("One! Two? Three."),
            ["One!", "Two?", "Three."],
        )

    def test_no_split_for_single_sentence(self):
        self.assertEqual(
            split_text_on_punctuation("Just one sentence."),
            ["Just one sentence."],
        )

    # --- Diversifier with split_on_punctuation=True ---

    def test_returns_one_result_per_original(self):
        div = Diversifier(methods=["echo"])
        results = div.diversify(
            ["One. Two!", "Single sentence"],
            n_styles=1,
            split_on_punctuation=True,
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["original"], "One. Two!")
        self.assertEqual(results[1]["original"], "Single sentence")

    def test_segments_are_reassembled_in_paraphrase(self):
        # PrefixMethod returns "<prefix>:<text>:<i>" per segment.
        # With two segments and n_styles=1, the paraphrase should be the
        # two segment paraphrases joined with a space.
        div = Diversifier(methods=[PrefixMethod("p")])
        results = div.diversify("One. Two!", n_styles=1, split_on_punctuation=True)
        self.assertEqual(results[0]["paraphrases"], ["p:One.:0 p:Two!:0"])

    def test_csv_returns_one_row_per_original(self):
        div = Diversifier(methods=["echo"])
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "bios.csv"
            pd.DataFrame({
                "id": [10, 20],
                "bio": ["First sentence. Second sentence!", "Only one."],
            }).to_csv(input_path, index=False)

            results = div.diversify(
                str(input_path), text_column="bio", n_styles=1, split_on_punctuation=True
            )
            self.assertIsInstance(results, pd.DataFrame)
            self.assertEqual(len(results), 2)
            self.assertIn("style 1", results.columns)


if __name__ == "__main__":
    unittest.main()
