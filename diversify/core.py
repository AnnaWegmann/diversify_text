"""
Core module for text diversification via stylistic paraphrasing.

Provides the :class:`Diversifier` class and a convenience :func:`diversify`
function that produce multiple stylistically varied paraphrases for each
input text.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import inspect
from pathlib import Path
import re
from typing import Any, Union
import warnings

import pandas as pd

from diversify.method import (
    DEFAULT_METHOD_REGISTRY,
    DiversificationMethod,
    MethodRegistry,
)

TextInput = Union[str, list[str], pd.Series, pd.DataFrame]
DiversifyOutput = Union[list[dict], pd.DataFrame]


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
        fallback_method: str | DiversificationMethod = "echo",
        method_registry: MethodRegistry | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._method_registry = method_registry or DEFAULT_METHOD_REGISTRY
        if methods is None:
            methods = [model_name or "tinystyler"]
        self._validate_registered_methods(methods)
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
        batch_size: int | None = None,
        split_on_punctuation: bool = False,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        method_kwargs: Mapping[str, dict[str, Any]] | None = None,
        output_path: str | Path | None = None,
    ) -> DiversifyOutput:
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
        batch_size : int, optional
            Number of texts to process per generation batch. If omitted,
            all texts are processed in a single batch.
        split_on_punctuation : bool
            If True, split each input text into punctuation-delimited
            segments before running methods.
        max_new_tokens : int
            Maximum number of tokens to generate per paraphrase.
        temperature : float
            Sampling temperature.
        top_p : float
            Nucleus-sampling probability mass.
        method_kwargs : mapping[str, dict], optional
            Per-method keyword arguments. Example:
            ``{"tinystyler": {"style_bank": [...]}}``.
        output_path : str | Path, optional
            When *texts* is a CSV/TSV filepath, save diversified output to this
            location. If omitted, defaults to
            ``<input_stem>_diversified<original_suffix>``.

        Returns
        -------
        list[dict] | pd.DataFrame
            For non-DataFrame input, returns a list with one entry per input text::

                {"original": str, "paraphrases": list[str]}

            For DataFrame input, returns a copy of the input DataFrame with
            added columns ``style 1`` .. ``style n``.
        """
        if n_styles < 1:
            raise ValueError("n_styles must be >= 1.")

        loaded_file = self._load_tabular_input(texts, text_column)
        source_df: pd.DataFrame | None = None
        input_path: Path | None = None
        input_sep: str | None = None
        non_tabular_original_ids: list[int] | None = None
        if loaded_file is not None:
            source_df, input_path, input_sep = loaded_file
            if split_on_punctuation:
                source_df = self._split_tabular_by_punctuation(source_df, text_column)
            text_list = source_df[text_column].tolist()
        else:
            base_texts = self._normalize_input(texts, text_column)
            if split_on_punctuation:
                text_list, non_tabular_original_ids = self._split_non_tabular_texts(
                    base_texts
                )
            else:
                text_list = base_texts

        method_kwargs = method_kwargs or {}
        if batch_size is None:
            effective_batch_size = len(text_list) if text_list else 1
        elif batch_size < 1:
            raise ValueError("batch_size must be >= 1 when provided.")
        else:
            effective_batch_size = batch_size

        paraphrases_by_text: list[list[str]] = []
        for start in range(0, len(text_list), effective_batch_size):
            batch_texts = text_list[start : start + effective_batch_size]
            batch_paraphrases = self._diversify_batch(
                batch_texts=batch_texts,
                n_styles=n_styles,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                method_kwargs=method_kwargs,
            )
            paraphrases_by_text.extend(batch_paraphrases)

        if isinstance(texts, pd.DataFrame) or source_df is not None:
            output_df = (source_df if source_df is not None else texts).copy()
            for idx in range(n_styles):
                output_df[f"style {idx + 1}"] = [
                    row[idx] if idx < len(row) else "" for row in paraphrases_by_text
                ]
            if input_path is not None:
                final_output_path = (
                    Path(output_path)
                    if output_path is not None
                    else input_path.with_name(
                        f"{input_path.stem}_diversified{input_path.suffix}"
                    )
                )
                final_output_path.parent.mkdir(parents=True, exist_ok=True)
                output_df.to_csv(
                    final_output_path,
                    sep=input_sep or ",",
                    index=False,
                )
            return output_df

        results = []
        for idx, (original, paraphrases) in enumerate(
            zip(text_list, paraphrases_by_text)
        ):
            item = {"original": original, "paraphrases": paraphrases}
            if non_tabular_original_ids is not None:
                item["original_id"] = non_tabular_original_ids[idx]
            results.append(item)
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

    def _load_tabular_input(
        self, texts: TextInput, text_column: str
    ) -> tuple[pd.DataFrame, Path, str] | None:
        """Load CSV/TSV input when *texts* points to a supported file path."""
        if not isinstance(texts, str):
            return None
        path = Path(texts)
        suffix = path.suffix.lower()
        if suffix not in {".csv", ".tsv"} or not path.is_file():
            return None
        sep = "," if suffix == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep)
        if text_column not in df.columns:
            available = ", ".join(df.columns)
            raise ValueError(
                f"Column '{text_column}' not found in {path}. Available: {available}"
            )
        df[text_column] = df[text_column].fillna("").astype(str).tolist()
        return df, path, sep

    @staticmethod
    def _split_text_on_punctuation(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?;:])\s+", text.strip())
        cleaned = [part.strip() for part in parts if part and part.strip()]
        return cleaned or [text.strip()]

    def _split_non_tabular_texts(self, texts: list[str]) -> tuple[list[str], list[int]]:
        split_texts: list[str] = []
        original_ids: list[int] = []
        for original_id, text in enumerate(texts):
            for segment in self._split_text_on_punctuation(text):
                split_texts.append(segment)
                original_ids.append(original_id)
        return split_texts, original_ids

    def _split_tabular_by_punctuation(
        self, df: pd.DataFrame, text_column: str
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        has_id = "id" in df.columns
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            row_original_id = row_dict["id"] if has_id else idx
            text = str(row_dict.get(text_column, "")).strip()
            segments = self._split_text_on_punctuation(text)
            for segment_idx, segment in enumerate(segments):
                new_row = dict(row_dict)
                new_row[text_column] = segment
                new_row["original_id"] = row_original_id
                new_row["segment_id"] = segment_idx
                rows.append(new_row)
        return pd.DataFrame(rows)

    def _resolve_method(
        self, method: str | DiversificationMethod
    ) -> DiversificationMethod:
        if isinstance(method, DiversificationMethod):
            return method
        if isinstance(method, str):
            method_cls = self._method_registry.get(method)
            init_kwargs = self._build_method_init_kwargs(method_cls)
            return method_cls(**init_kwargs)
        raise TypeError("method must be str or DiversificationMethod instance.")

    def _build_method_init_kwargs(
        self, method_cls: type[DiversificationMethod]
    ) -> dict[str, Any]:
        """Pass only constructor kwargs supported by the target method class."""
        signature = inspect.signature(method_cls)
        init_kwargs: dict[str, Any] = {}
        if "device" in signature.parameters:
            init_kwargs["device"] = self.device
        if "model_name" in signature.parameters:
            init_kwargs["model_name"] = self.model_name
        return init_kwargs

    def _resolve_methods(
        self, methods: Sequence[str | DiversificationMethod]
    ) -> list[DiversificationMethod]:
        resolved = [self._resolve_method(a) for a in methods]
        if not resolved:
            raise ValueError("At least one method is required.")
        return resolved

    def _validate_registered_methods(
        self, methods: Sequence[str | DiversificationMethod]
    ) -> None:
        missing = sorted(
            {m for m in methods if isinstance(m, str) and m not in self._method_registry}
        )
        if missing:
            available = self._method_registry.names()
            raise KeyError(
                f"Unknown methods: {', '.join(missing)}. "
                f"Available: {', '.join(available)}"
            )

    @staticmethod
    def _compute_allocations(total_styles: int, n_methods: int) -> list[int]:
        base, remainder = divmod(total_styles, n_methods)
        return [base + (1 if i < remainder else 0) for i in range(n_methods)]

    def _diversify_batch(
        self,
        *,
        batch_texts: list[str],
        n_styles: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        method_kwargs: Mapping[str, dict[str, Any]],
    ) -> list[list[str]]:
        allocations = self._compute_allocations(n_styles, len(self._methods))
        paraphrases_by_text = [[] for _ in batch_texts]

        styles_generated = 0
        for method, allocated_styles in zip(self._methods, allocations):
            if allocated_styles <= 0:
                continue
            kwargs = method_kwargs.get(method.name, {})
            try:
                partial = method.generate(
                    batch_texts,
                    n_styles=allocated_styles,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **kwargs,
                )
                generated_styles = self._merge_paraphrases(
                    paraphrases_by_text, partial, batch_texts
                )
                styles_generated += generated_styles
            except Exception as exc:
                warnings.warn(
                    f"Method '{method.name}' failed and fallback will be used: "
                    f"{type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        missing_styles = n_styles - styles_generated
        if missing_styles > 0:
            fallback_partial = self._fallback_method.generate(
                batch_texts,
                n_styles=missing_styles,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            self._merge_paraphrases(paraphrases_by_text, fallback_partial, batch_texts)

        return paraphrases_by_text

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
    fallback_method: str | DiversificationMethod = "echo",
    method_registry: MethodRegistry | None = None,
    **kwargs,
) -> DiversifyOutput:
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
    fallback_method : str | DiversificationMethod
        Fallback used when methods fail.
    method_registry : MethodRegistry, optional
        Custom registry for method name resolution.
    **kwargs
        Forwarded to :meth:`Diversifier.diversify`
        (``n_styles``, ``text_column``, ``batch_size``,
        ``split_on_punctuation``, ``max_new_tokens``,
        ``temperature``, ``top_p``, ``method_kwargs``, ``output_path``).

    Returns
    -------
    list[dict] | pd.DataFrame
        See :meth:`Diversifier.diversify`.
    """
    div = Diversifier(
        model_name=model_name,
        device=device,
        methods=methods,
        fallback_method=fallback_method,
        method_registry=method_registry,
    )
    return div.diversify(texts, **kwargs)
