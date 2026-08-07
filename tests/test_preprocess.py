"""Tests for sentence splitting and the split_on_punctuation=True behaviour
in Diversifier."""

import json
import tempfile
import unittest
from pathlib import Path

from diversify_text import Diversifier
from diversify_text._preprocess import split_sentences
from tests.fixtures import PrefixMethod


class TestSentenceSplitting(unittest.TestCase):

    # --- split_sentences ---

    def test_splits_on_period(self):
        self.assertEqual(
            split_sentences("First. Second."),
            ["First.", "Second."],
        )

    def test_splits_on_multiple_punctuation_types(self):
        self.assertEqual(
            split_sentences("One! Two? Three."),
            ["One!", "Two?", "Three."],
        )

    def test_no_split_for_single_sentence(self):
        self.assertEqual(
            split_sentences("Just one sentence."),
            ["Just one sentence."],
        )

    def test_handles_abbreviations(self):
        self.assertEqual(
            split_sentences("Dr. Smith went home. He was tired."),
            ["Dr. Smith went home.", "He was tired."],
        )

    def test_handles_decimals(self):
        self.assertEqual(
            split_sentences("He scored 3.5 points. That was good."),
            ["He scored 3.5 points.", "That was good."],
        )

    def test_empty_string(self):
        self.assertEqual(
            split_sentences(""),
            [""],
        )

    # --- Diversifier with split_on_punctuation=True ---

    def test_returns_one_result_per_original(self):
        div = Diversifier(method="echo")
        results = div.diversify(
            ["One. Two!", "Single sentence"],
            n=1,
            preprocess_kwargs={"split_on_punctuation": True},
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["original"], "One. Two!")
        self.assertEqual(results[1]["original"], "Single sentence")

    def test_segments_are_reassembled_in_paraphrase(self):
        div = Diversifier(method=PrefixMethod("p"))
        results = div.diversify(
            "One. Two!", n=1,
            preprocess_kwargs={"split_on_punctuation": True},
        )
        self.assertEqual(
            results[0]["paraphrases"],
            [{"style": "informal_tinystyler", "text": "p:One.:0 p:Two!:0"}],
        )

    def test_csv_with_punctuation_writes_one_jsonl_record_per_original(self):
        div = Diversifier(method="echo")
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "tmp.csv"
            input_path.write_text(
                "id,bio\n10,First sentence. Second sentence!\n20,Only one.\n",
                encoding="utf-8",
            )

            result = div.diversify(
                str(input_path), text_column="bio", n=1,
                preprocess_kwargs={"split_on_punctuation": True},
            )
            self.assertIsInstance(result, Path)
            jsonl_path = Path(tmpdir) / "tmp_diversified.jsonl"
            lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 2)
            record = json.loads(lines[0])
            self.assertEqual(record["original"], "First sentence. Second sentence!")


if __name__ == "__main__":
    unittest.main()
