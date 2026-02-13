"""
Core module for text diversification via stylistic paraphrasing.

Provides the :class:`Diversifier` class and a convenience :func:`diversify`
function that produce multiple stylistically varied paraphrases for each
input text.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Union
import warnings

import pandas as pd

from diversify.method import (
    DEFAULT_METHOD_REGISTRY,
    DiversificationMethod,
    MethodRegistry,
)

TextInput = Union[str, list[str], pd.Series, pd.DataFrame]


class Diversifier:
    """Generate stylistic paraphrases using one or more pluggable methods.

    Each method can be a separate model or algorithm. The class supports
    combining many methods and automatically distributing requested styles
    across them.

    Parameters
    ----------
    model_name : str, optional
        Backwards-compatible alias for selecting a single method by name.
        If ``None``, defaults to ``"tinystyler"``.
    device : str, optional
        Torch device (``"cuda"``, ``"cpu"``, ``"mps"``, ...).
    methods : sequence[str | DiversificationMethod], optional
        Method names and/or pre-built method instances.
    strict_methods : bool
        If ``True``, fail fast when a method errors. If ``False``,
        failed methods are replaced with fallback generation.
    fallback_method : str | DiversificationMethod
        Fallback method used to fill missing outputs.

    Example
    -------
    >>> div = Diversifier(methods=["tinystyler", "echo"])
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
        methods: Sequence[str | DiversificationMethod] | None = None,
        strict_methods: bool = False,
        fallback_method: str | DiversificationMethod = "echo",
        method_registry: MethodRegistry | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.strict_methods = strict_methods
        self._method_registry = method_registry or DEFAULT_METHOD_REGISTRY
        if methods is None:
            methods = [model_name or "tinystyler"]
        self._methods = self._resolve_methods(methods)
        self._fallback_method = self._resolve_method(fallback_method)

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
        method_kwargs: Mapping[str, dict[str, Any]] | None = None,
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
        method_kwargs : mapping[str, dict], optional
            Per-method keyword arguments. Example:
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
        allocations = self._compute_allocations(n_styles, len(self._methods))
        paraphrases_by_text = [[] for _ in text_list]
        method_kwargs = method_kwargs or {}

        styles_generated = 0
        for method, allocated_styles in zip(self._methods, allocations):
            if allocated_styles <= 0:
                continue
            kwargs = method_kwargs.get(method.name, {})
            try:
                partial = method.generate(
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
            except Exception as exc:
                if self.strict_methods:
                    raise
                warnings.warn(
                    f"Method '{method.name}' failed and fallback will be used: "
                    f"{type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        missing_styles = n_styles - styles_generated
        if missing_styles > 0:
            fallback_partial = self._fallback_method.generate(
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

    def _resolve_method(
        self, method: str | DiversificationMethod
    ) -> DiversificationMethod:
        if isinstance(method, DiversificationMethod):
            return method
        if isinstance(method, str):
            return self._method_registry.create(
                method,
                device=self.device,
                model_name=self.model_name,
            )
        raise TypeError("method must be str or DiversificationMethod instance.")

    def _resolve_methods(
        self, methods: Sequence[str | DiversificationMethod]
    ) -> list[DiversificationMethod]:
        resolved = [self._resolve_method(a) for a in methods]
        if not resolved:
            raise ValueError("At least one method is required.")
        return resolved

    @staticmethod
    def _compute_allocations(total_styles: int, n_methods: int) -> list[int]:
        base, remainder = divmod(total_styles, n_methods)
        return [base + (1 if i < remainder else 0) for i in range(n_methods)]

    @staticmethod
    def _merge_paraphrases(
        combined: list[list[str]],
        incoming: list[list[str]],
        source_texts: list[str],
    ) -> int:
        if len(incoming) != len(source_texts):
            raise ValueError("Method returned invalid batch size.")
        generated_styles: int | None = None
        for idx, group in enumerate(incoming):
            if not isinstance(group, list) or not all(
                isinstance(item, str) for item in group
            ):
                raise TypeError("Method output must be list[list[str]].")
            if generated_styles is None:
                generated_styles = len(group)
            elif len(group) != generated_styles:
                raise ValueError(
                    "Method must return the same number of styles for each text."
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
    methods: Sequence[str | DiversificationMethod] | None = None,
    strict_methods: bool = False,
    fallback_method: str | DiversificationMethod = "echo",
    method_registry: MethodRegistry | None = None,
    **kwargs,
) -> list[dict]:
    """One-shot convenience function: create a :class:`Diversifier` and run it.

    Parameters
    ----------
    texts : str | list[str] | pd.Series | pd.DataFrame
        Input text(s).
    model_name : str, optional
        Backwards-compatible alias for selecting a single method by name.
    device : str, optional
        Torch device.
    methods : sequence[str | DiversificationMethod], optional
        Method names and/or pre-built method instances.
    strict_methods : bool
        If True, raise if a method fails.
    fallback_method : str | DiversificationMethod
        Fallback used when methods fail (when strict_methods is False).
    method_registry : MethodRegistry, optional
        Custom registry for method name resolution.
    **kwargs
        Forwarded to :meth:`Diversifier.diversify`
        (``n_styles``, ``text_column``, ``max_new_tokens``,
        ``temperature``, ``top_p``, ``method_kwargs``).

    Returns
    -------
    list[dict]
        See :meth:`Diversifier.diversify`.
    """
    div = Diversifier(
        model_name=model_name,
        device=device,
        methods=methods,
        strict_methods=strict_methods,
        fallback_method=fallback_method,
        method_registry=method_registry,
    )
    return div.diversify(texts, **kwargs)
