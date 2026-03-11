"""MIS-based content preservation filter.

Uses the `Mutual Implication Score <https://huggingface.co/s-nlp/Mutual_Implication_Score>`_
to score paraphrases against their originals and identify those that fall below
a content-preservation threshold.
"""

from __future__ import annotations

import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)

_DEFAULT_MIN_SCORE = 0.80
_DEFAULT_N_CANDIDATES = 5


class MISFilter:
    """Score paraphrases against originals using Mutual Implication Score.

    The model is loaded lazily on first use.

    Parameters
    ----------
    device : str, optional
        Torch device for the MIS model (e.g. ``"cpu"``, ``"cuda"``).
    min_score : float
        Minimum acceptable MIS score.  Paraphrases scoring below this
        are considered failures.
    n_candidates : int
        Number of alternative paraphrases to generate for each
        below-threshold paraphrase.  The best-scoring candidate
        replaces the original.
    """

    def __init__(
        self,
        device: str | None = None,
        *,
        min_score: float = _DEFAULT_MIN_SCORE,
        n_candidates: int = _DEFAULT_N_CANDIDATES,
    ) -> None:
        self.device = device or "cpu"
        self.min_score = min_score
        self.n_candidates = n_candidates
        self._mis = None

    def prepare(self) -> None:
        """Pre-load the MIS model."""
        self._ensure_model()

    def _ensure_model(self):
        if self._mis is None:
            from mutual_implication_score import MIS

            logger.info("Loading MIS model on %s ...", self.device)
            self._mis = MIS(device=self.device)
            logger.info("MIS model loaded.")
        return self._mis

    def score(
        self,
        originals: list[str],
        paraphrases: list[str],
    ) -> list[float]:
        """Compute MIS scores for aligned (original, paraphrase) pairs.

        Parameters
        ----------
        originals : list[str]
            Source texts, one per pair.
        paraphrases : list[str]
            Generated paraphrases, aligned 1:1 with *originals*.

        Returns
        -------
        list[float]
            MIS scores in [0, 1], one per pair.
        """
        mis = self._ensure_model()
        return mis.compute(originals, paraphrases)

    def score_batch(
        self,
        originals: list[str],
        paraphrases_by_text: list[list[str]],
    ) -> list[list[float]]:
        """Score all paraphrases for a batch of originals.

        Flattens the nested structure, scores in one MIS call, then
        reshapes back.

        Returns
        -------
        list[list[float]]
            Scores shaped ``[n_texts][n_styles]``.
        """
        flat_originals: list[str] = []
        flat_paraphrases: list[str] = []
        sizes: list[int] = []
        for orig, paras in zip(originals, paraphrases_by_text):
            for p in paras:
                flat_originals.append(orig)
                flat_paraphrases.append(p)
            sizes.append(len(paras))

        total = len(flat_originals)
        if total == 0:
            return [[] for _ in originals]

        logger.info("Scoring %d paraphrase(s) with MIS ...", total)
        flat_scores = self.score(flat_originals, flat_paraphrases)

        result: list[list[float]] = []
        idx = 0
        for size in sizes:
            result.append(flat_scores[idx : idx + size])
            idx += size
        return result

    def identify_failures(
        self,
        scores_by_text: list[list[float]],
    ) -> list[list[int]]:
        """Return indices of paraphrases below *min_score* for each text.

        Returns
        -------
        list[list[int]]
            For each text, a list of style indices that failed.
        """
        return [
            [i for i, s in enumerate(scores) if s < self.min_score]
            for scores in scores_by_text
        ]
