"""Tests for evaluation plumbing (diversify_text._evaluate, _metrics).

Uses stub metrics throughout, so nothing is downloaded and no model is
loaded.  The real metrics are covered in test_metrics.py.
"""

import json
import tempfile
import unittest
from pathlib import Path

from diversify_text import diversify, evaluate
from diversify_text._metrics import METRIC_REGISTRY, Metric, resolve_metrics


# ------------------------------------------------------------------
# Stub metrics
# ------------------------------------------------------------------


class LengthRatio(Metric):
    """Paraphrase length over original length — a metric with known values."""

    name = "length_ratio"

    def compute(self, originals, paraphrases):
        return {self.name: [len(p) / len(o) for o, p in zip(originals, paraphrases)]}


class DualScore(Metric):
    """Two selectable sub-scores, like rouge."""

    name = "dual"
    variants = ("a", "b")
    default_variants = ("a",)

    def compute(self, originals, paraphrases):
        both = {"a": [1.0] * len(originals), "b": [2.0] * len(originals)}
        return {v: both[v] for v in self.selected}


class ModelBacked(Metric):
    """Takes a model, so model handling can be tested without downloads."""

    name = "model_backed"
    accepts_model = True
    default_model = "default-model"

    def compute(self, originals, paraphrases):
        offset = 0.0 if self.model == self.default_model else 10.0
        return {self.name: [offset + len(p) for p in paraphrases]}


# Length ratios: 0.5 and 1.5 for the first text, 2.0 and 1.0 for the
# second — so text means are 1.0 and 1.5, and the dataset mean is 1.25.
RECORDS = [
    {
        "original": "abcd",
        "paraphrases": [
            {"style": "informal", "text": "ab"},
            {"style": "formal", "text": "abcdef"},
        ],
    },
    {
        "original": "xy",
        "paraphrases": [
            {"style": "informal", "text": "xyxy"},
            {"style": "formal", "text": "xy"},
        ],
    },
]


class TestGranularity(unittest.TestCase):

    def test_dataset_is_the_default(self):
        result = evaluate(RECORDS, metrics=[LengthRatio()])
        self.assertEqual(result.dataset["length_ratio"], 1.25)
        self.assertIsNone(result.pair)
        self.assertIsNone(result.text)

    def test_text_level_averages_each_text(self):
        result = evaluate(RECORDS, metrics=[LengthRatio()], granularity="text")
        self.assertEqual([r["length_ratio"] for r in result.text], [1.0, 1.5])
        self.assertEqual(result.text[0]["n_paraphrases"], 2)

    def test_pair_level_keeps_order_and_style(self):
        result = evaluate(RECORDS, metrics=[LengthRatio()], granularity="pair")
        self.assertEqual(len(result.pair), 4)
        self.assertEqual(
            [p["style"] for p in result.pair],
            ["informal", "formal", "informal", "formal"],
        )
        self.assertEqual([p["text_index"] for p in result.pair], [0, 0, 1, 1])
        self.assertEqual(result.pair[0]["length_ratio"], 0.5)

    def test_all_populates_every_level(self):
        result = evaluate(RECORDS, metrics=[LengthRatio()], granularity="all")
        self.assertIsNotNone(result.pair)
        self.assertIsNotNone(result.text)
        self.assertIsNotNone(result.dataset)

    def test_all_agrees_with_single_granularity_runs(self):
        every = evaluate(RECORDS, metrics=[LengthRatio()], granularity="all")
        single = evaluate(RECORDS, metrics=[LengthRatio()], granularity="dataset")
        self.assertEqual(every.dataset, single.dataset)

    def test_levels_are_consistent_with_each_other(self):
        pair = evaluate(RECORDS, metrics=[LengthRatio()], granularity="pair")
        text = evaluate(RECORDS, metrics=[LengthRatio()], granularity="text")
        dataset = evaluate(RECORDS, metrics=[LengthRatio()], granularity="dataset")

        first = [p["length_ratio"] for p in pair.pair if p["text_index"] == 0]
        self.assertAlmostEqual(text.text[0]["length_ratio"], sum(first) / len(first))

        every = [p["length_ratio"] for p in pair.pair]
        self.assertAlmostEqual(dataset.dataset["length_ratio"], sum(every) / len(every))


class TestMetricSelection(unittest.TestCase):

    def test_default_variants_are_used(self):
        result = evaluate(RECORDS, metrics=[DualScore()])
        self.assertEqual(set(result.dataset), {"a"})

    def test_variants_select_columns(self):
        result = evaluate(RECORDS, metrics=[DualScore(variants=["b"])])
        self.assertEqual(set(result.dataset), {"b"})
        self.assertEqual(result.dataset["b"], 2.0)

    def test_variants_via_registry_name(self):
        METRIC_REGISTRY["dual"] = DualScore
        try:
            result = evaluate(
                RECORDS,
                metrics=["dual"],
                metric_kwargs={"dual": {"variants": ["a", "b"]}},
            )
            self.assertEqual(set(result.dataset), {"a", "b"})
        finally:
            del METRIC_REGISTRY["dual"]

    def test_multiple_metrics_merge_their_columns(self):
        result = evaluate(
            RECORDS, metrics=[LengthRatio(), DualScore()], granularity="pair"
        )
        self.assertIn("length_ratio", result.pair[0])
        self.assertIn("a", result.pair[0])

    def test_model_argument_reaches_the_metric(self):
        default = evaluate(RECORDS, metrics=[ModelBacked()])
        custom = evaluate(RECORDS, metrics=[ModelBacked(model="other")])
        self.assertNotEqual(
            default.dataset["model_backed"], custom.dataset["model_backed"]
        )

    def test_model_argument_via_metric_kwargs(self):
        METRIC_REGISTRY["model_backed"] = ModelBacked
        try:
            result = evaluate(
                RECORDS,
                metrics=["model_backed"],
                metric_kwargs={"model_backed": {"model": "other"}},
            )
            self.assertGreater(result.dataset["model_backed"], 10.0)
        finally:
            del METRIC_REGISTRY["model_backed"]

    def test_plain_string_paraphrases_are_accepted(self):
        records = [{"original": "abcd", "paraphrases": ["ab", "abcdef"]}]
        result = evaluate(records, metrics=[LengthRatio()])
        self.assertEqual(result.dataset["length_ratio"], 1.0)


class TestErrors(unittest.TestCase):

    def test_unknown_metric_name(self):
        with self.assertRaises(ValueError):
            evaluate(RECORDS, metrics=["does_not_exist"])

    def test_unknown_granularity(self):
        with self.assertRaises(ValueError):
            evaluate(RECORDS, metrics=[LengthRatio()], granularity="sentence")

    def test_model_rejected_by_fixed_model_metric(self):
        with self.assertRaises(ValueError):
            resolve_metrics(["rouge"], {"rouge": {"model": "x"}})

    def test_non_string_model_rejected(self):
        with self.assertRaises(ValueError):
            resolve_metrics(["bertscore"], {"bertscore": {"model": 42}})

    def test_variants_rejected_by_single_score_metric(self):
        with self.assertRaises(ValueError):
            resolve_metrics(["chrf"], {"chrf": {"variants": ["a"]}})

    def test_unknown_variant(self):
        with self.assertRaises(ValueError):
            resolve_metrics(["rouge"], {"rouge": {"variants": ["rouge9"]}})

    def test_unapplied_metric_kwargs(self):
        with self.assertRaises(ValueError):
            resolve_metrics(["rouge"], {"chrf": {}})

    def test_metric_kwargs_for_a_prebuilt_instance(self):
        with self.assertRaises(ValueError):
            resolve_metrics([DualScore()], {"dual": {"variants": ["b"]}})

    def test_empty_output(self):
        with self.assertRaises(ValueError):
            evaluate([], metrics=[LengthRatio()])

    def test_record_missing_keys(self):
        with self.assertRaises(ValueError):
            evaluate([{"paraphrases": []}], metrics=[LengthRatio()])

    def test_no_paraphrases_to_score(self):
        with self.assertRaises(ValueError):
            evaluate([{"original": "x", "paraphrases": []}], metrics=[LengthRatio()])

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            evaluate(Path("/nonexistent/results.jsonl"), metrics=[LengthRatio()])

    def test_unsupported_input_type(self):
        with self.assertRaises(TypeError):
            evaluate(42, metrics=[LengthRatio()])


class TestJSONL(unittest.TestCase):

    def test_evaluating_a_path_matches_the_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "diversified.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in RECORDS), encoding="utf-8")

            from_disk = evaluate(path, metrics=[LengthRatio()])
            in_memory = evaluate(RECORDS, metrics=[LengthRatio()])
            self.assertEqual(from_disk.dataset, in_memory.dataset)

    def test_to_jsonl_tags_every_row_with_its_level(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = evaluate(RECORDS, metrics=[LengthRatio()], granularity="all")
            out = result.to_jsonl(Path(tmpdir) / "eval.jsonl")

            lines = out.read_text(encoding="utf-8").strip().split("\n")
            levels = [json.loads(line)["level"] for line in lines]
            self.assertEqual(levels.count("pair"), 4)
            self.assertEqual(levels.count("text"), 2)
            self.assertEqual(levels.count("dataset"), 1)

    def test_pair_rows_carry_their_texts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = evaluate(
                RECORDS, metrics=[LengthRatio()], granularity="pair"
            ).to_jsonl(Path(tmpdir) / "eval.jsonl")

            row = json.loads(out.read_text(encoding="utf-8").split("\n")[0])
            self.assertEqual(row["original"], "abcd")
            self.assertEqual(row["paraphrase"], "ab")
            self.assertEqual(row["style"], "informal")

    def test_to_jsonl_creates_missing_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = evaluate(RECORDS, metrics=[LengthRatio()]).to_jsonl(
                Path(tmpdir) / "nested" / "eval.jsonl"
            )
            self.assertTrue(out.exists())


class TestCallForms(unittest.TestCase):
    """``diversify(...).evaluate()`` must equal ``evaluate(diversify(...))``."""

    def test_output_is_still_a_list_of_records(self):
        results = diversify("hello", n=2, method="echo")
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        self.assertEqual(set(results[0]), {"original", "paraphrases"})

    def test_output_records_its_device(self):
        results = diversify("hello", n=2, method="echo")
        self.assertTrue(hasattr(results, "device"))

    def test_method_and_function_agree(self):
        results = diversify("hello", n=2, method="echo")
        self.assertEqual(
            results.evaluate(metrics=[LengthRatio()]).dataset,
            evaluate(results, metrics=[LengthRatio()]).dataset,
        )

    def test_file_output_has_no_evaluate_method(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = diversify("hello", n=2, method="echo", output_dir=tmpdir)
            self.assertIsInstance(result, Path)
            self.assertFalse(hasattr(result, "evaluate"))

    def test_file_output_can_still_be_evaluated_by_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = diversify("hello", n=2, method="echo", output_dir=tmpdir)
            result = evaluate(path, metrics=[LengthRatio()])
            self.assertIn("length_ratio", result.dataset)


class TestRegistry(unittest.TestCase):

    def test_registry_holds_the_documented_metrics(self):
        self.assertEqual(
            set(METRIC_REGISTRY),
            {"style_similarity", "bertscore", "rouge", "chrf", "mis"},
        )

    def test_none_selects_every_metric(self):
        self.assertEqual(len(resolve_metrics(None)), len(METRIC_REGISTRY))

    def test_prebuilt_instances_are_used_as_given(self):
        instance = LengthRatio()
        self.assertIs(resolve_metrics([instance])[0], instance)

    def test_device_reaches_metrics_built_by_name(self):
        self.assertEqual(resolve_metrics(["chrf"], device="cuda")[0].device, "cuda")

    def test_prebuilt_instances_keep_their_own_device(self):
        instance = LengthRatio(device="cpu")
        self.assertEqual(resolve_metrics([instance], device="cuda")[0].device, "cpu")

    def test_non_metric_entry_raises(self):
        with self.assertRaises(TypeError):
            resolve_metrics([42])


if __name__ == "__main__":
    unittest.main()