"""Tests for the real metrics on real sentences.

These load models.  Classes skip themselves when their dependency is
missing, so the suite still runs on a machine without them::

    pip install "diversify-text[eval]"

Two environment variables control the heavier runs:

``DIVERSIFY_TEST_DEVICE``
    Device for the model-based metrics (default ``cpu``).
``DIVERSIFY_TEST_MODELS``
    Set to ``1`` to run the metrics that download large weights
    (BERTScore, MIS).  They are skipped by default.
"""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

from diversify_text import evaluate
from diversify_text._metrics import clear_metric_cache

DEVICE = os.environ.get("DIVERSIFY_TEST_DEVICE", "cpu")


SOURCE = "The experiment was conducted in a controlled lab setting."

#: Paraphrases of decreasing faithfulness to ``SOURCE``.  Every metric
#: that measures *content* should rank them in this order.
LADDER = [
    ("identical", SOURCE),
    ("close", "The experiment was carried out in a controlled laboratory setting."),
    ("loose", "They ran the study indoors, under conditions they kept tightly controlled."),
    ("unrelated", "The cat knocked a mug off the kitchen table this morning."),
]

RECORDS = [
    {
        "original": SOURCE,
        "paraphrases": [{"style": s, "text": t} for s, t in LADDER],
    },
    {
        "original": "She graduated from MIT in 2019.",
        "paraphrases": [
            {"style": "identical", "text": "She graduated from MIT in 2019."},
            {"style": "close", "text": "She finished her degree at MIT back in 2019."},
            {"style": "loose", "text": "MIT is where she wrapped up her studies, in 2019."},
            {"style": "unrelated", "text": "The train was forty minutes late again."},
        ],
    },
]


def installed(*modules):
    """True when every named module can be imported."""
    return all(importlib.util.find_spec(m) is not None for m in modules)


class ContentMetricMixin:
    """Shared checks for metrics that measure content similarity.

    Subclasses set ``metric`` and, optionally, ``ceilings`` — the score
    an exact copy should reach for each column.

    The ordering checks are what catch a swapped ``originals`` /
    ``paraphrases`` argument: MIS takes ``(source, paraphrase)`` while
    BERTScore takes ``(predictions, references)`` the other way round.
    """

    metric = ""
    ceilings: dict = {}

    @classmethod
    def setUpClass(cls):
        # Score once per class — model loading dominates the runtime.
        cls.result = evaluate(
            RECORDS, metrics=[cls.metric], granularity="pair", device=DEVICE
        )
        cls.by_style = {
            row["style"]: {c: row[c] for c in cls.result.metrics}
            for row in cls.result.pair
        }

    def test_identical_scores_highest(self):
        for column in self.result.metrics:
            values = [s[column] for s in self.by_style.values()]
            self.assertEqual(
                self.by_style["identical"][column], max(values),
                f"{column}: an exact copy did not score highest — "
                f"originals and paraphrases may be swapped",
            )

    def test_unrelated_scores_lowest(self):
        for column in self.result.metrics:
            values = [s[column] for s in self.by_style.values()]
            self.assertEqual(self.by_style["unrelated"][column], min(values), column)

    def test_close_beats_loose(self):
        for column in self.result.metrics:
            self.assertGreater(
                self.by_style["close"][column],
                self.by_style["loose"][column],
                column,
            )

    def test_exact_match_reaches_the_ceiling(self):
        for column, ceiling in self.ceilings.items():
            self.assertAlmostEqual(
                self.by_style["identical"][column], ceiling, places=4, msg=column
            )

    def test_one_score_per_pair(self):
        expected = sum(len(r["paraphrases"]) for r in RECORDS)
        self.assertEqual(len(self.result.pair), expected)
        for column in self.result.metrics:
            self.assertEqual(
                sum(1 for row in self.result.pair if column in row), expected
            )

    def test_scores_are_finite_floats(self):
        for row in self.result.pair:
            for column in self.result.metrics:
                value = row[column]
                self.assertIsInstance(value, float)
                self.assertEqual(value, value, f"{column} produced NaN")
                self.assertGreaterEqual(value, 0.0, column)


@unittest.skipUnless(installed("sacrebleu"), "sacrebleu not installed")
class TestChrF(ContentMetricMixin, unittest.TestCase):
    metric = "chrf"
    ceilings = {"chrf": 100.0}


@unittest.skipUnless(installed("evaluate"), "evaluate not installed")
class TestRouge(ContentMetricMixin, unittest.TestCase):
    metric = "rouge"
    ceilings = {"rouge1": 1.0, "rouge2": 1.0, "rougeL": 1.0}

    def test_default_variants(self):
        self.assertEqual(set(self.result.metrics), {"rouge1", "rouge2", "rougeL"})

    def test_variants_select_columns(self):
        result = evaluate(
            RECORDS, metrics=["rouge"], device=DEVICE,
            metric_kwargs={"rouge": {"variants": ["rouge1", "rougeLsum"]}},
        )
        self.assertEqual(set(result.dataset), {"rouge1", "rougeLsum"})

    def test_selecting_a_variant_does_not_change_its_value(self):
        picked = evaluate(
            RECORDS, metrics=["rouge"], device=DEVICE,
            metric_kwargs={"rouge": {"variants": ["rouge1"]}},
        )
        every = evaluate(RECORDS, metrics=["rouge"], device=DEVICE)
        self.assertAlmostEqual(
            picked.dataset["rouge1"], every.dataset["rouge1"], places=6
        )


@unittest.skipUnless(
    installed("evaluate", "bert_score"), "bert_score not installed"
)
class TestBertScore(ContentMetricMixin, unittest.TestCase):
    metric = "bertscore"

    def test_defaults_to_f1_only(self):
        self.assertEqual(set(self.result.metrics), {"bertscore_f1"})

    def test_variants_select_columns(self):
        result = evaluate(
            RECORDS, metrics=["bertscore"], device=DEVICE,
            metric_kwargs={"bertscore": {"variants": ["precision", "recall", "f1"]}},
        )
        self.assertEqual(
            set(result.dataset),
            {"bertscore_precision", "bertscore_recall", "bertscore_f1"},
        )


@unittest.skipUnless(
    installed("mutual_implication_score"), "mutual_implication_score not installed"
)
class TestMIS(ContentMetricMixin, unittest.TestCase):
    metric = "mis"


@unittest.skipUnless(
    installed("sentence_transformers"), "sentence_transformers not installed"
)
class TestStyleSimilarity(unittest.TestCase):
    """Style embeddings measure *how* a text is written, not what it says.

    An unrelated sentence in the same register can legitimately outscore
    a faithful paraphrase, so the ordering checks in
    :class:`ContentMetricMixin` do not apply here.
    """

    @classmethod
    def setUpClass(cls):
        cls.result = evaluate(
            RECORDS, metrics=["style_similarity"], granularity="pair", device=DEVICE
        )

    def test_scores_are_cosine_similarities(self):
        for row in self.result.pair:
            self.assertGreaterEqual(row["style_similarity"], -1.0)
            self.assertLessEqual(row["style_similarity"], 1.0001)

    def test_identical_texts_are_near_one(self):
        identical = next(
            r for r in self.result.pair if r["style"] == "identical"
        )
        self.assertAlmostEqual(identical["style_similarity"], 1.0, places=3)

    def test_a_different_model_gives_different_scores(self):
        other = evaluate(
            RECORDS, metrics=["style_similarity"], device=DEVICE,
            metric_kwargs={
                "style_similarity": {"model": "sentence-transformers/all-MiniLM-L6-v2"}
            },
        )
        default = evaluate(RECORDS, metrics=["style_similarity"], device=DEVICE)
        self.assertNotAlmostEqual(
            other.dataset["style_similarity"],
            default.dataset["style_similarity"],
            places=4,
        )


@unittest.skipUnless(
    installed("sentence_transformers"), "sentence_transformers not installed"
)
class TestModelCache(unittest.TestCase):

    def setUp(self):
        clear_metric_cache()

    def tearDown(self):
        clear_metric_cache()

    def test_model_is_cached_and_reused(self):
        from diversify_text._metrics import _MODEL_CACHE

        evaluate(RECORDS, metrics=["style_similarity"], device=DEVICE)
        self.assertGreater(len(_MODEL_CACHE), 0)

        before = dict(_MODEL_CACHE)
        evaluate(RECORDS, metrics=["style_similarity"], device=DEVICE)
        for key, model in before.items():
            self.assertIs(
                _MODEL_CACHE[key], model,
                "a second run reloaded the model instead of reusing it",
            )

    def test_a_different_model_gets_its_own_entry(self):
        from diversify_text._metrics import _MODEL_CACHE

        evaluate(RECORDS, metrics=["style_similarity"], device=DEVICE)
        evaluate(
            RECORDS, metrics=["style_similarity"], device=DEVICE,
            metric_kwargs={
                "style_similarity": {"model": "sentence-transformers/all-MiniLM-L6-v2"}
            },
        )
        self.assertEqual(len(_MODEL_CACHE), 2)

    def test_clearing_the_cache_empties_it(self):
        from diversify_text._metrics import _MODEL_CACHE

        evaluate(RECORDS, metrics=["style_similarity"], device=DEVICE)
        clear_metric_cache()
        self.assertEqual(len(_MODEL_CACHE), 0)


@unittest.skipUnless(installed("sacrebleu"), "sacrebleu not installed")
class TestMetricsTogether(unittest.TestCase):
    """Metrics run together must give the same numbers as run alone."""

    metrics = ["chrf"] + (["rouge"] if installed("evaluate") else [])

    def test_combined_run_matches_individual_runs(self):
        together = evaluate(RECORDS, metrics=self.metrics, device=DEVICE).dataset
        alone = {}
        for name in self.metrics:
            alone.update(evaluate(RECORDS, metrics=[name], device=DEVICE).dataset)

        self.assertEqual(set(together), set(alone))
        for column, value in alone.items():
            self.assertAlmostEqual(together[column], value, places=6, msg=column)

    def test_levels_are_consistent(self):
        pair = evaluate(RECORDS, metrics=self.metrics, granularity="pair",
                        device=DEVICE)
        dataset = evaluate(RECORDS, metrics=self.metrics, granularity="dataset",
                           device=DEVICE)
        for column in pair.metrics:
            values = [row[column] for row in pair.pair]
            self.assertAlmostEqual(
                dataset.dataset[column], sum(values) / len(values), places=6
            )

    def test_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = evaluate(
                RECORDS, metrics=self.metrics, granularity="all", device=DEVICE
            ).to_jsonl(Path(tmpdir) / "eval.jsonl")

            lines = out.read_text(encoding="utf-8").strip().split("\n")
            levels = [json.loads(line)["level"] for line in lines]
            self.assertEqual(levels.count("pair"), 8)
            self.assertEqual(levels.count("text"), 2)
            self.assertEqual(levels.count("dataset"), 1)


if __name__ == "__main__":
    unittest.main()