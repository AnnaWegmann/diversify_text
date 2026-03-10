"""
Core module for text diversification via stylistic paraphrasing.

Provides the :class:`Diversifier` class and a convenience :func:`diversify`
function that produce multiple stylistically varied paraphrases for each
input text.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import inspect
import logging
from itertools import islice
from pathlib import Path
from typing import Any

from tqdm import tqdm

from diversify._input import TextInput, resolve_input
from diversify._output import DiversifyOutput, OutputWriter, resolve_output_path
from diversify._text import split_text_on_punctuation
from diversify.method import DEFAULT_METHOD_REGISTRY, DiversificationMethod

logger = logging.getLogger(__name__)


class Diversifier:
    """Generate stylistic paraphrases using one or more pluggable methods.

    Each method can be a separate model or algorithm. The class supports
    combining many methods and automatically distributing requested styles
    across them.

    Parameters
    ----------
    device : str, optional
        Torch device (``"cuda"``, ``"cpu"``, ``"mps"``, ...).
    methods : sequence[str | DiversificationMethod], optional
        Method names and/or pre-built method instances.

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
        device: str | None = None,
        *,
        methods: Sequence[str | DiversificationMethod] | None = None,
    ) -> None:
        self.device = device
        if methods is None:
            methods = ["tinystyler"]
        self._validate_registered_methods(methods)
        self._methods = self._resolve_methods(methods)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diversify(
        self,
        texts: TextInput,
        *,
        n_styles: int = 5,
        text_column: str = "text",
        batch_size: int = 32,
        split_on_punctuation: bool = False,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int = 51173,
        method_kwargs: Mapping[str, dict[str, Any]] | None = None,
        output_dir: str | Path | None = None,
        output_name: str | None = None,
    ) -> DiversifyOutput:
        """Produce *n_styles* stylistic paraphrases for each input text.

        Parameters
        ----------
        texts : str | list[str] | Iterable[str]
            A single text, a list of texts, a generator/iterable of texts,
            or a path to a ``.csv``, ``.tsv``, or ``.txt`` file.
        n_styles : int
            Number of stylistically diverse paraphrases to generate per
            input text.
        text_column : str
            Column name to extract when *texts* points to a CSV/TSV file.
        batch_size : int
            Number of texts to pull from the input iterator per batch.
        split_on_punctuation : bool
            If True, split each input text into punctuation-delimited
            segments before running methods.
        max_new_tokens : int
            Maximum number of tokens to generate per paraphrase.
        temperature : float
            Sampling temperature.
        top_p : float
            Nucleus-sampling probability mass.
        seed : int
            Random seed for reproducible output.  Defaults to ``51173``.
            Pass a different integer to get a new set of outputs, or
            ``None`` to disable seeding (non-deterministic).
        method_kwargs : mapping[str, dict], optional
            Per-method keyword arguments. Example:
            ``{"tinystyler": {"style_bank": [...]}}``.
        output_dir : str | Path, optional
            Directory to write output files into.  When provided for
            ``str`` / ``list[str]`` input, forces disk output instead of
            in-memory.  Defaults vary by input type (see
            :func:`resolve_output_path`).
        output_name : str, optional
            Base filename (without extension).  The ``.jsonl`` extension
            is appended automatically.

        Returns
        -------
        list[dict] | Path
            For in-memory input (``str``, ``list[str]``) without
            *output_dir*, returns a list with one entry per input text::

                {"original": str, "paraphrases": list[str]}

            Otherwise, returns the ``Path`` to the output file(s).
        """
        if n_styles < 1:
            raise ValueError("n_styles must be >= 1.")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")

        # --- resolve input & output ---
        text_iter, input_context = resolve_input(texts, text_column)
        out_path = resolve_output_path(input_context, output_dir, output_name)

        # --- prepare methods (model loading etc.) ---
        method_kwargs = method_kwargs or {}
        for method in self._methods:
            method.prepare()

        if seed is not None:
            import torch
            torch.manual_seed(seed)
            logger.info("Using random seed: %d", seed)

        # --- process batches lazily ---
        writer = OutputWriter(input_context, n_styles, out_path)
        writer.open()
        try:
            with tqdm(total=input_context.total, desc="Diversifying", unit="text") as pbar:
                while True:
                    batch_texts = list(islice(text_iter, batch_size))
                    if not batch_texts:
                        break

                    if split_on_punctuation:
                        segments_per_text = [
                            split_text_on_punctuation(t) for t in batch_texts
                        ]
                        flat_segments = [
                            seg for segs in segments_per_text for seg in segs
                        ]
                        paraphrases_by_segment = self._diversify_batch(
                            batch_texts=flat_segments,
                            n_styles=n_styles,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            method_kwargs=method_kwargs,
                        )
                        paraphrases_by_text = self._reassemble_from_segments(
                            segments_per_text, paraphrases_by_segment, n_styles
                        )
                    else:
                        paraphrases_by_text = self._diversify_batch(
                            batch_texts=batch_texts,
                            n_styles=n_styles,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            method_kwargs=method_kwargs,
                        )

                    writer.write_batch(batch_texts, paraphrases_by_text)
                    pbar.update(len(batch_texts))
        except Exception:
            writer.finish()
            raise

        return writer.finish()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_method(
        self, method: str | DiversificationMethod
    ) -> DiversificationMethod:
        if isinstance(method, DiversificationMethod):
            return method
        if isinstance(method, str):
            method_cls = DEFAULT_METHOD_REGISTRY.get(method)
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
            {m for m in methods if isinstance(m, str) and m not in DEFAULT_METHOD_REGISTRY}
        )
        if missing:
            available = DEFAULT_METHOD_REGISTRY.names()
            raise KeyError(
                f"Unknown methods: {', '.join(missing)}. "
                f"Available: {', '.join(available)}"
            )

    @staticmethod
    def _reassemble_from_segments(
        segments_per_text: list[list[str]],
        paraphrases_by_segment: list[list[str]],
        n_styles: int,
    ) -> list[list[str]]:
        """Join per-segment paraphrases back into per-original-text paraphrases."""
        result = []
        seg_idx = 0
        for segs in segments_per_text:
            seg_paras = paraphrases_by_segment[seg_idx : seg_idx + len(segs)]
            result.append([" ".join(sp[i] for sp in seg_paras) for i in range(n_styles)])
            seg_idx += len(segs)
        return result

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

        for method, allocated_styles in zip(self._methods, allocations):
            if allocated_styles <= 0:
                continue
            kwargs = method_kwargs.get(method.name, {})
            partial = method.generate(
                batch_texts,
                n_styles=allocated_styles,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )
            self._merge_paraphrases(paraphrases_by_text, partial, batch_texts)

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
    device: str | None = None,
    methods: Sequence[str | DiversificationMethod] | None = None,
    **kwargs,
) -> DiversifyOutput:
    """One-shot convenience function: create a :class:`Diversifier` and run it.

    Parameters
    ----------
    texts : str | list[str] | Iterable[str]
        Input text(s).
    device : str, optional
        Torch device.
    methods : sequence[str | DiversificationMethod], optional
        Method names and/or pre-built method instances.
    **kwargs
        Forwarded to :meth:`Diversifier.diversify`
        (``n_styles``, ``text_column``, ``batch_size``,
        ``split_on_punctuation``, ``max_new_tokens``,
        ``temperature``, ``top_p``, ``seed``,
        ``method_kwargs``, ``output_dir``, ``output_name``).

    Returns
    -------
    list[dict] | Path
        See :meth:`Diversifier.diversify`.
    """
    div = Diversifier(device=device, methods=methods)
    return div.diversify(texts, **kwargs)
