"""Tests for the MIS content preservation filter."""

import unittest
from unittest.mock import MagicMock

from diversify_text import Diversifier
from diversify_text.filter.mis import MISFilter

from tests.fixtures import IndexedMethod


class TestMISFilterUnit(unittest.TestCase):
    """Unit tests for MISFilter in isolation."""

    def setUp(self):
        self.f = MISFilter(min_score=0.80)

    def test_score_batch_reshapes_correctly(self):
        self.f.score = MagicMock(
            return_value=[0.9, 0.5, 0.8, 0.7, 0.6, 0.95]
        )

        scores = self.f.score_batch(
            ["text1", "text2"],
            [["p1a", "p1b", "p1c"], ["p2a", "p2b", "p2c"]],
        )
        self.assertEqual(scores, [[0.9, 0.5, 0.8], [0.7, 0.6, 0.95]])

    def test_identify_failures(self):
        failures = self.f.identify_failures([[0.9, 0.5, 0.8], [0.7, 0.6, 0.95]])
        self.assertEqual(failures, [[1], [0, 1]])

    def test_identify_failures_none_below(self):
        failures = self.f.identify_failures([[0.9, 0.85]])
        self.assertEqual(failures, [[]])

    def test_score_batch_empty(self):
        scores = self.f.score_batch(["text"], [[]])
        self.assertEqual(scores, [[]])


class TestMISFilterIntegration(unittest.TestCase):
    """Integration tests: MISFilter wired into Diversifier."""

    def _make_diversifier(self, **kwargs):
        """Create a Diversifier with IndexedMethod and a mocked MIS scorer."""
        kwargs.setdefault("method", IndexedMethod())
        kwargs.setdefault("semantic_filter", True)
        div = Diversifier(**kwargs)
        self._mock_score = MagicMock()
        div._mis_filter.score = self._mock_score
        return div

    def test_no_filter_by_default(self):
        div = Diversifier(method="echo")
        self.assertIsNone(div._mis_filter)

    def test_constructor_stores_params(self):
        div = Diversifier(
            method="echo",
            semantic_filter=True,
            min_score=0.90,
            n_candidates=3,
        )
        self.assertIsNotNone(div._mis_filter)
        self.assertEqual(div._mis_filter.min_score, 0.90)
        self.assertEqual(div._mis_filter.n_candidates, 3)

    def test_selects_best_candidate_for_failed_style(self):
        div = self._make_diversifier(min_score=0.80, n_candidates=3)
        # n=2, n_candidates=3.
        # IndexedMethod produces: c0 → ["hello:s0:c0", "hello:s1:c0"]
        #                         c1 → ["hello:s0:c1", "hello:s1:c1"]
        #                         c2 → ["hello:s0:c2", "hello:s1:c2"]
        self._mock_score.side_effect = [
            [0.5, 0.9],    # candidate 0: style 0 fails
            [0.85, 0.92],  # candidate 1: style 0 passes
            [0.7, 0.88],   # candidate 2: style 0 passes
        ]

        results = div.diversify("hello", n=2)
        paraphrases = results[0]["paraphrases"]
        self.assertEqual(len(paraphrases), 2)
        # Style 0: first candidate failed (0.5), best is candidate 1 (0.85).
        self.assertEqual(paraphrases[0]["text"], "hello:s0:c1")
        # Style 1: first candidate passed (0.9), kept as-is.
        self.assertEqual(paraphrases[1]["text"], "hello:s1:c0")
        # 3 score calls: first candidate + 2 remaining.
        self.assertEqual(self._mock_score.call_count, 3)

    def test_keeps_best_even_if_all_below_threshold(self):
        div = self._make_diversifier(min_score=0.80, n_candidates=3)
        self._mock_score.side_effect = [
            [0.3],   # candidate 0: fails
            [0.6],   # candidate 1: still below
            [0.5],   # candidate 2: still below
        ]

        results = div.diversify("hello", n=1)
        paraphrases = results[0]["paraphrases"]
        self.assertEqual(len(paraphrases), 1)
        # Best score is 0.6 from candidate 1.
        self.assertEqual(paraphrases[0]["text"], "hello:s0:c1")


if __name__ == "__main__":
    unittest.main()
