"""MIS-based content preservation filter.

Uses the `Mutual Implication Score <https://huggingface.co/s-nlp/Mutual_Implication_Score>`_
to score paraphrases against their originals and identify those that fall below
a content-preservation threshold.
"""

from __future__ import annotations

import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)

_DEFAULT_MIN_SCORE = 0.70
_DEFAULT_N_CANDIDATES = 3


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

            from diversify._utils import suppress_hf_load_noise

            with suppress_hf_load_noise():
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

        Re-implementation of ``MIS.compute()`` from
        `mutual_implication_score <https://github.com/s-nlp/mutual_implication_score/blob/b47c88e978f510b3dee1d4bde1f22b054c67ad62/mutual_implication_score/mis_wrapper.py#L38>`_.
        The upstream version crashes on NumPy ≥ 2.0 because
        ``model()`` returns shape ``(batch, 1)`` tensors, and after
        ``.cpu().numpy()`` + ``list.extend()`` each element is a 1-D
        array of shape ``(1,)`` on which ``float()`` raises
        ``TypeError: only 0-dimensional arrays can be converted to
        Python scalars``.

        Fix: added ``.flatten()`` before ``.tolist()`` so the tensor is
        squeezed to 1-D before conversion.

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
        import torch
        from torch.utils.data import DataLoader

        mis = self._ensure_model()

        # Follows the structure of MIS.compute():
        # https://github.com/s-nlp/mutual_implication_score/blob/b47c88e978f510b3dee1d4bde1f22b054c67ad62/mutual_implication_score/mis_wrapper.py#L38
        #
        # Changes from upstream:
        # - PairsDatasetInference replaced with list(zip(...)) (equivalent)
        # - merged_prob uses .flatten().tolist() instead of .cpu().numpy()
        #   followed by float(e), which crashes on NumPy ≥ 2.0.
        dataset_direct = list(zip(originals, paraphrases))
        dataloader_direct = DataLoader(dataset_direct, batch_size=16)
        dataset_reverse = list(zip(paraphrases, originals))
        dataloader_reverse = DataLoader(dataset_reverse, batch_size=16)

        preds = []
        for b1, b2 in zip(dataloader_direct, dataloader_reverse):
            with torch.no_grad():
                tokenized1 = mis.tokenizer(
                    *b1, padding=True, truncation="longest_first",
                    return_tensors="pt",
                ).to(mis.device)
                tokenized2 = mis.tokenizer(
                    *b2, padding=True, truncation="longest_first",
                    return_tensors="pt",
                ).to(mis.device)
                merged_prob = mis.model(tokenized1, tokenized2)
                merged_prob = torch.sigmoid(merged_prob)
            # upstream: merged_prob.cpu().numpy() then float(e) — crashes on
            # NumPy ≥ 2.0 because elements are shape-(1,) arrays.
            preds.extend(merged_prob.cpu().flatten().tolist())
        return preds

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
