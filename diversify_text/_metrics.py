"""Evaluation metrics for diversify.

Every metric scores ``(original, paraphrase)`` pairs and returns
``dict[str, list[float]]`` — one list of per-pair scores per reported
sub-score.  Single-score metrics return one entry keyed by their name;
BERTScore and ROUGE return one entry per selected variant.

Heavy dependencies are imported inside :meth:`Metric.prepare` so that
``sentence_transformers``, ``evaluate``, ``sacrebleu`` and
``mutual_implication_score`` stay optional.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

_log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Model cache
# ------------------------------------------------------------------

_MODEL_CACHE: dict[tuple[str, str | None, str], Any] = {}


def _cached(key: tuple[str, str | None, str], loader: Callable[[], Any]) -> Any:
    """Return a cached model, loading it on first use."""
    if key not in _MODEL_CACHE:
        _log.info("Loading model for metric %r (model=%r, device=%r)", *key)
        _MODEL_CACHE[key] = loader()
    return _MODEL_CACHE[key]


def clear_metric_cache() -> None:
    """Drop cached metric models so memory can be reclaimed."""
    _MODEL_CACHE.clear()


# ------------------------------------------------------------------
# Base class
# ------------------------------------------------------------------


class Metric:
    """Base class for evaluation metrics.

    Subclasses set the class attributes below and implement
    :meth:`prepare` and :meth:`compute`.

    Attributes
    ----------
    name : str
        Registry name, also the result key for single-score metrics.
    accepts_model : bool
        Whether the metric takes a ``model`` argument.  Passing one to a
        metric that does not raises ``ValueError``.
    default_model : str or None
        Model used when the caller does not pass one.
    variants : tuple[str, ...]
        Sub-scores the metric can report.  Empty means the metric
        reports a single score and rejects ``variants``.
    default_variants : tuple[str, ...]
        Sub-scores reported when the caller does not select any.
    """

    name: str = ""
    accepts_model: bool = False
    default_model: str | None = None
    variants: tuple[str, ...] = ()
    default_variants: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        device: str = "cpu",
        model: str | None = None,
        variants: list[str] | None = None,
    ) -> None:
        self.device = device

        if model is not None and not self.accepts_model:
            raise ValueError(
                f"Metric {self.name!r} has a fixed model and does not "
                f"accept a 'model' argument."
            )
        if model is not None and not isinstance(model, str):
            raise ValueError(
                f"Metric {self.name!r} expects 'model' as a string "
                f"identifier, got {type(model).__name__}."
            )
        self.model = model if model is not None else self.default_model

        if variants is not None and not self.variants:
            raise ValueError(
                f"Metric {self.name!r} reports a single score and does "
                f"not accept a 'variants' argument."
            )
        selected = tuple(variants) if variants is not None else self.default_variants
        unknown = [v for v in selected if v not in self.variants]
        if unknown:
            raise ValueError(
                f"Unknown variants for metric {self.name!r}: {unknown}. "
                f"Available: {list(self.variants)}."
            )
        if self.variants and not selected:
            raise ValueError(f"Metric {self.name!r} needs at least one variant.")
        self.selected = selected

    # --- lifecycle ---

    def prepare(self) -> None:
        """Import dependencies and load models.  Called once before use."""

    def compute(
        self,
        originals: list[str],
        paraphrases: list[str],
    ) -> dict[str, list[float]]:
        """Score each ``(original, paraphrase)`` pair.

        Returns one list of per-pair scores per reported sub-score;
        every list has the same length as the inputs.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r}, device={self.device!r})"


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------


class StyleSimilarity(Metric):
    """Cosine similarity between sentence-embedding representations.

    With the default model this measures *style* similarity rather than
    semantic similarity.
    """

    name = "style_similarity"
    accepts_model = True
    default_model = "AnnaWegmann/Style-Embedding"

    def prepare(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._encoder = _cached(
            (self.name, self.model, self.device),
            lambda: SentenceTransformer(self.model, device=self.device),
        )

    def compute(self, originals, paraphrases):
        import numpy as np

        a = self._encoder.encode(originals, convert_to_numpy=True)
        b = self._encoder.encode(paraphrases, convert_to_numpy=True)
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
        return {self.name: [float(s) for s in (a * b).sum(axis=1)]}


class BertScore(Metric):
    """BERTScore precision, recall and F1.

    All three come from the same forward pass; ``variants`` only
    controls what is reported.
    """

    name = "bertscore"
    accepts_model = True
    default_model = None  # None → the default model for `lang`
    variants = ("precision", "recall", "f1")
    default_variants = ("f1",)

    def __init__(self, *, lang: str = "en", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.lang = lang

    def prepare(self) -> None:
        import evaluate as hf_evaluate

        self._metric = _cached(
            (self.name, None, "-"),
            lambda: hf_evaluate.load("bertscore"),
        )

    def compute(self, originals, paraphrases):
        kwargs: dict[str, Any] = {
            "predictions": paraphrases,
            "references": originals,
            "lang": self.lang,
            "device": self.device,
        }
        if self.model is not None:
            kwargs["model_type"] = self.model
        result = self._metric.compute(**kwargs)
        return {
            f"bertscore_{v}": [float(x) for x in result[v]] for v in self.selected
        }


class Rouge(Metric):
    """ROUGE n-gram and longest-common-subsequence overlap."""

    name = "rouge"
    variants = ("rouge1", "rouge2", "rougeL", "rougeLsum")
    default_variants = ("rouge1", "rouge2", "rougeL")

    def prepare(self) -> None:
        import evaluate as hf_evaluate

        self._metric = _cached(
            (self.name, None, "-"),
            lambda: hf_evaluate.load("rouge"),
        )

    def compute(self, originals, paraphrases):
        result = self._metric.compute(
            predictions=paraphrases,
            references=originals,
            rouge_types=list(self.selected),
            use_aggregator=False,
        )
        return {v: [float(x) for x in result[v]] for v in self.selected}


class ChrF(Metric):
    """chrF++ — character n-gram F-score with word bigrams."""

    name = "chrf"

    def prepare(self) -> None:
        from sacrebleu.metrics import CHRF

        self._scorer = _cached(
            (self.name, None, "-"),
            lambda: CHRF(word_order=2),
        )

    def compute(self, originals, paraphrases):
        scores = [
            float(self._scorer.sentence_score(para, [orig]).score)
            for orig, para in zip(originals, paraphrases)
        ]
        return {self.name: scores}


class Mis(Metric):
    """Mutual Implication Score — symmetric semantic similarity.

    Uses the fixed model shipped with ``mutual_implication_score``.
    """

    name = "mis"

    def prepare(self) -> None:
        from mutual_implication_score import MIS

        self._scorer = _cached(
            (self.name, None, self.device),
            lambda: MIS(device=self.device),
        )

    def compute(self, originals, paraphrases):
        scores = self._scorer.compute(originals, paraphrases)
        return {self.name: [float(s) for s in scores]}


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

METRIC_REGISTRY: dict[str, type[Metric]] = {
    cls.name: cls for cls in (StyleSimilarity, BertScore, Rouge, ChrF, Mis)
}


def resolve_metrics(
    metrics: list[str | Metric] | None = None,
    metric_kwargs: dict[str, dict[str, Any]] | None = None,
    device: str = "cpu",
) -> list[Metric]:
    """Turn metric names and keyword arguments into metric instances.

    Parameters
    ----------
    metrics : list of str or Metric, optional
        Names from :data:`METRIC_REGISTRY` and/or pre-built instances.
        ``None`` selects every registered metric.
    metric_kwargs : dict, optional
        Per-metric keyword arguments, keyed by metric name.  Entries
        that are not applied raise ``ValueError``.
    device : str
        Device passed to every metric built here.  Pre-built instances
        keep their own device.

    Returns
    -------
    list[Metric]
    """
    remaining = dict(metric_kwargs or {})
    names: list[str | Metric] = (
        list(METRIC_REGISTRY) if metrics is None else list(metrics)
    )

    resolved: list[Metric] = []
    for entry in names:
        if isinstance(entry, Metric):
            resolved.append(entry)
        elif isinstance(entry, str):
            if entry not in METRIC_REGISTRY:
                raise ValueError(
                    f"Unknown metric {entry!r}. "
                    f"Available: {sorted(METRIC_REGISTRY)}."
                )
            resolved.append(
                METRIC_REGISTRY[entry](device=device, **remaining.pop(entry, {}))
            )
        else:
            raise TypeError(
                f"metrics entries must be a name or a Metric, got "
                f"{type(entry).__name__}."
            )

    if remaining:
        raise ValueError(
            f"metric_kwargs entries were not applied: {sorted(remaining)}. "
            f"They name a metric that was not selected, is unknown, or was "
            f"passed as a pre-built instance."
        )
    return resolved