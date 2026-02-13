"""
Core module for text diversification via stylistic paraphrasing.

Provides the :class:`Diversifier` class and a convenience :func:`diversify`
function that produce multiple stylistically varied paraphrases for each
input text.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Union

import pandas as pd

from diversify.method import (
    DEFAULT_APPROACH_REGISTRY,
    ApproachRegistry,
    DiversificationApproach,
)

TextInput = Union[str, list[str], pd.Series, pd.DataFrame]


class Diversifier:
    """Generate stylistic paraphrases using one or more pluggable approaches.

    Each approach can be a separate model or algorithm. The class supports
    combining many approaches and automatically distributing requested styles
    across them.

    Parameters
    ----------
    model_name : str, optional
        Backwards-compatible alias for selecting a single approach by name.
        If ``None``, defaults to ``"tinystyler"``.
    device : str, optional
        Torch device (``"cuda"``, ``"cpu"``, ``"mps"``, ...).
    approaches : sequence[str | DiversificationApproach], optional
        Approach names and/or pre-built approach instances.
    strict_approaches : bool
        If ``True``, fail fast when an approach errors. If ``False``,
        failed approaches are replaced with fallback generation.
    fallback_approach : str | DiversificationApproach
        Fallback approach used to fill missing outputs.

    Example
    -------
    >>> div = Diversifier(approaches=["tinystyler", "echo"])
    >>> results = div.diversify("The experiment was conducted in a lab.")
    >>> len(results)  # one dict per input text
    1
    >>> list(results[0].keys())
    ['original', 'paraphrases']
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        *,
        approaches: Sequence[str | DiversificationApproach] | None = None,
        strict_approaches: bool = False,
        fallback_approach: str | DiversificationApproach = "echo",
        approach_registry: ApproachRegistry | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.strict_approaches = strict_approaches
        self._approach_registry = approach_registry or DEFAULT_APPROACH_REGISTRY
        if approaches is None:
            approaches = [model_name or "tinystyler"]
        self._approaches = self._resolve_approaches(approaches)
        self._fallback_approach = self._resolve_approach(fallback_approach)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diversify(
        self,
        texts: TextInput,
        *,
        n_styles: int = 5,
        text_column: str = "text",
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        approach_kwargs: Mapping[str, dict[str, Any]] | None = None,
    ) -> list[dict]:
        """Produce *n_styles* stylistic paraphrases for each input text.

        Parameters
        ----------
        texts : str | list[str] | pd.Series | pd.DataFrame
            Input text(s).  When a :class:`~pandas.DataFrame` is passed,
            the column specified by *text_column* is used.
        n_styles : int
            Number of stylistically diverse paraphrases to generate per
            input text.
        text_column : str
            Column name to read when *texts* is a DataFrame.
        max_new_tokens : int
            Maximum number of tokens to generate per paraphrase.
        temperature : float
            Sampling temperature.
        top_p : float
            Nucleus-sampling probability mass.
        approach_kwargs : mapping[str, dict], optional
            Per-approach keyword arguments. Example:
            ``{"tinystyler": {"style_bank": [...]}}``.

        Returns
        -------
        list[dict]
            A list with one entry per input text.  Each entry is a dict::

                {
                    "original": str,
                    "paraphrases": list[str],   # length == n_styles
                }
        """
        if n_styles < 1:
            raise ValueError("n_styles must be >= 1.")

        text_list = self._normalize_input(texts, text_column)
        allocations = self._compute_allocations(n_styles, len(self._approaches))
        paraphrases_by_text = [[] for _ in text_list]
        approach_kwargs = approach_kwargs or {}

        styles_generated = 0
        for approach, allocated_styles in zip(self._approaches, allocations):
            if allocated_styles <= 0:
                continue
            kwargs = approach_kwargs.get(approach.name, {})
            try:
                partial = approach.generate(
                    text_list,
                    n_styles=allocated_styles,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **kwargs,
                )
                generated_styles = self._merge_paraphrases(
                    paraphrases_by_text, partial, text_list
                )
                styles_generated += generated_styles
            except Exception:
                if self.strict_approaches:
                    raise

        missing_styles = n_styles - styles_generated
        if missing_styles > 0:
            fallback_partial = self._fallback_approach.generate(
                text_list,
                n_styles=missing_styles,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            self._merge_paraphrases(paraphrases_by_text, fallback_partial, text_list)

        results = []
        for original, paraphrases in zip(text_list, paraphrases_by_text):
            results.append({"original": original, "paraphrases": paraphrases})

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_input(self, texts: TextInput, text_column: str) -> list[str]:
        """Coerce any supported input type into ``list[str]``."""
        if isinstance(texts, str):
            return [texts]
        if isinstance(texts, pd.Series):
            return texts.tolist()
        if isinstance(texts, pd.DataFrame):
            return texts[text_column].tolist()
        if isinstance(texts, list):
            return texts
        raise TypeError(
            f"Unsupported input type {type(texts).__name__}. "
            "Expected str, list[str], pd.Series, or pd.DataFrame."
        )

    def _resolve_approach(
        self, approach: str | DiversificationApproach
    ) -> DiversificationApproach:
        if isinstance(approach, DiversificationApproach):
            return approach
        if isinstance(approach, str):
            return self._approach_registry.create(
                approach,
                device=self.device,
                model_name=self.model_name,
            )
        raise TypeError("approach must be str or DiversificationApproach instance.")

    def _resolve_approaches(
        self, approaches: Sequence[str | DiversificationApproach]
    ) -> list[DiversificationApproach]:
        resolved = [self._resolve_approach(a) for a in approaches]
        if not resolved:
            raise ValueError("At least one approach is required.")
        return resolved

    @staticmethod
    def _compute_allocations(total_styles: int, n_approaches: int) -> list[int]:
        base, remainder = divmod(total_styles, n_approaches)
        return [base + (1 if i < remainder else 0) for i in range(n_approaches)]

    @staticmethod
    def _merge_paraphrases(
        combined: list[list[str]],
        incoming: list[list[str]],
        source_texts: list[str],
    ) -> int:
        if len(incoming) != len(source_texts):
            raise ValueError("Approach returned invalid batch size.")
        generated_styles: int | None = None
        for idx, group in enumerate(incoming):
            if not isinstance(group, list) or not all(
                isinstance(item, str) for item in group
            ):
                raise TypeError("Approach output must be list[list[str]].")
            if generated_styles is None:
                generated_styles = len(group)
            elif len(group) != generated_styles:
                raise ValueError(
                    "Approach must return the same number of styles for each text."
                )
            combined[idx].extend(group)
        return generated_styles or 0


# ------------------------------------------------------------------
# Module-level convenience function
# ------------------------------------------------------------------


def diversify(
    texts: TextInput,
    *,
    model_name: str | None = None,
    device: str | None = None,
    approaches: Sequence[str | DiversificationApproach] | None = None,
    strict_approaches: bool = False,
    fallback_approach: str | DiversificationApproach = "echo",
    approach_registry: ApproachRegistry | None = None,
    **kwargs,
) -> list[dict]:
    """One-shot convenience function: create a :class:`Diversifier` and run it.

    Parameters
    ----------
    texts : str | list[str] | pd.Series | pd.DataFrame
        Input text(s).
    model_name : str, optional
        Backwards-compatible alias for selecting a single approach by name.
    device : str, optional
        Torch device.
    approaches : sequence[str | DiversificationApproach], optional
        Approach names and/or pre-built approach instances.
    strict_approaches : bool
        If True, raise if an approach fails.
    fallback_approach : str | DiversificationApproach
        Fallback used when approaches fail (when strict_approaches is False).
    approach_registry : ApproachRegistry, optional
        Custom registry for approach name resolution.
    **kwargs
        Forwarded to :meth:`Diversifier.diversify`
        (``n_styles``, ``text_column``, ``max_new_tokens``,
        ``temperature``, ``top_p``, ``approach_kwargs``).

    Returns
    -------
    list[dict]
        See :meth:`Diversifier.diversify`.
    """
    div = Diversifier(
        model_name=model_name,
        device=device,
        approaches=approaches,
        strict_approaches=strict_approaches,
        fallback_approach=fallback_approach,
        approach_registry=approach_registry,
    )
    return div.diversify(texts, **kwargs)
