"""Evaluation of diversify output.

Scores each ``(original, paraphrase)`` pair with one or more metrics and
reports the result at pair, text and/or dataset level.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from diversify_text._metrics import Metric, resolve_metrics

_log = logging.getLogger(__name__)

GRANULARITIES = ("pair", "text", "dataset", "all")


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Scores at the requested granularity.

    Attributes
    ----------
    granularity : str
        The granularity that was requested.
    metrics : list[str]
        Names of the score columns, in report order.
    pair, text, dataset
        The available levels; the ones outside *granularity* are ``None``.
    """

    granularity: str
    metrics: list[str]
    pair: list[dict[str, Any]] | None = None
    text: list[dict[str, Any]] | None = None
    dataset: dict[str, float] | None = None

    def levels(self) -> list[tuple[str, list[dict[str, Any]]]]:
        """Return the populated levels as ``(name, rows)`` pairs."""
        out = []
        if self.pair is not None:
            out.append(("pair", self.pair))
        if self.text is not None:
            out.append(("text", self.text))
        if self.dataset is not None:
            out.append(("dataset", [self.dataset]))
        return out

    def to_jsonl(self, path: str | Path) -> Path:
        """Write one JSON object per line, each tagged with its level.

        Every row carries a ``"level"`` key, so a result with several
        levels stays readable in a single file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for level, rows in self.levels():
                for row in rows:
                    handle.write(
                        json.dumps({"level": level, **row}, ensure_ascii=False) + "\n"
                    )
        _log.info("Evaluation written to %s", path)
        return path

    def __repr__(self) -> str:
        if self.dataset:
            scores = ", ".join(f"{k}={v:.4f}" for k, v in self.dataset.items())
        else:
            scores = ", ".join(self.metrics)
        return f"EvaluationResult(granularity={self.granularity!r}, {scores})"


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def evaluate(
    output,
    *,
    metrics: list[str | Metric] | None = None,
    metric_kwargs: dict[str, dict[str, Any]] | None = None,
    granularity: str = "dataset",
    device: str | None = None,
) -> EvaluationResult:
    """Score diversify output against the original texts.

    Parameters
    ----------
    output : DiversifyResult | list[dict] | str | Path
        The value returned by :func:`diversify`, or a path to a
        ``.jsonl`` file it wrote.
    metrics : list of str or Metric, optional
        Which metrics to run.  ``None`` runs all of them:
        ``style_similarity``, ``bertscore``, ``rouge``, ``chrf``, ``mis``.
    metric_kwargs : dict, optional
        Per-metric keyword arguments keyed by metric name, e.g.
        ``{"bertscore": {"variants": ["precision", "f1"]}}``.
    granularity : {"pair", "text", "dataset", "all"}
        ``"pair"`` scores every original–paraphrase pair, ``"text"``
        averages over each input text's paraphrases, ``"dataset"``
        averages over everything, and ``"all"`` reports all three.
    device : str, optional
        Device for the model-based metrics.  Defaults to the device the
        output was generated on, and to ``"cpu"`` when that is unknown.

    Returns
    -------
    EvaluationResult

    Example
    -------
    >>> from diversify_text import diversify, evaluate
    >>> evaluate(diversify("The experiment was conducted in a lab."))
    """
    if granularity not in GRANULARITIES:
        raise ValueError(
            f"Unknown granularity {granularity!r}. Available: {list(GRANULARITIES)}."
        )

    records = _load_records(output)
    if device is None:
        device = getattr(output, "device", None) or "cpu"

    resolved = resolve_metrics(metrics, metric_kwargs, device=device)
    if not resolved:
        raise ValueError("No metrics selected.")

    # --- flatten every pair, so each metric runs once ---
    index, originals, paraphrases, styles = [], [], [], []
    for i, record in enumerate(records):
        for entry in record["paraphrases"]:
            index.append(i)
            originals.append(record["original"])
            paraphrases.append(entry["text"] if isinstance(entry, dict) else entry)
            styles.append(entry.get("style") if isinstance(entry, dict) else None)

    if not originals:
        raise ValueError("Nothing to evaluate: the output contains no paraphrases.")

    scores: dict[str, list[float]] = {}
    for metric in resolved:
        metric.prepare()
        for key, values in metric.compute(originals, paraphrases).items():
            if len(values) != len(originals):
                raise ValueError(
                    f"Metric {metric.name!r} returned {len(values)} scores "
                    f"for {len(originals)} pairs."
                )
            scores[key] = values

    columns = list(scores)
    want = GRANULARITIES if granularity == "all" else (granularity,)
    result = EvaluationResult(granularity=granularity, metrics=columns)

    if "pair" in want:
        result.pair = [
            {
                "text_index": index[i],
                "original": originals[i],
                "style": styles[i],
                "paraphrase": paraphrases[i],
                **{c: scores[c][i] for c in columns},
            }
            for i in range(len(originals))
        ]

    if "text" in want:
        rows = []
        for i, record in enumerate(records):
            positions = [j for j, t in enumerate(index) if t == i]
            rows.append(
                {
                    "text_index": i,
                    "original": record["original"],
                    "n_paraphrases": len(positions),
                    **{c: fmean(scores[c][j] for j in positions) for c in columns},
                }
            )
        result.text = rows

    if "dataset" in want:
        result.dataset = {c: fmean(scores[c]) for c in columns}

    return result


# ------------------------------------------------------------------
# Input handling
# ------------------------------------------------------------------


def _load_records(output) -> list[dict[str, Any]]:
    """Accept a diversify result or a path to the ``.jsonl`` it wrote."""
    if isinstance(output, (str, Path)):
        path = Path(output)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")
        with open(path, encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    elif isinstance(output, list):
        records = output
    else:
        raise TypeError(
            f"Expected diversify output or a path to its .jsonl file, got "
            f"{type(output).__name__}."
        )

    if not records:
        raise ValueError("Nothing to evaluate: the output is empty.")
    for record in records:
        if "original" not in record or "paraphrases" not in record:
            raise ValueError(
                "Each record needs 'original' and 'paraphrases' keys; got "
                f"{sorted(record)}."
            )
    return records