"""
Core module for text diversification via stylistic paraphrasing.

Provides the :class:`Diversifier` class and a convenience :func:`diversify`
function that produce multiple stylistically varied paraphrases for each
input text.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
from itertools import islice
from pathlib import Path
from typing import Any

from tqdm import tqdm

from diversify_text._input import TextInput, resolve_input
from diversify_text._output import DiversifyOutput, OutputWriter, resolve_output_path
from diversify_text._postprocess import postprocess
from diversify_text._preprocess import preprocess
import diversify_text._cache as _cache
from diversify_text.filter.mis import MISFilter
from diversify_text.method import DEFAULT_METHOD_REGISTRY, DiversificationMethod

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
        similarity_filter: bool = False,
        _methods: list[DiversificationMethod] | None = None,
        _mis_filter: MISFilter | None = None,
        **filter_kwargs: Any,
    ) -> None:
        self.device = device
        if _methods is not None:
            self._methods = _methods
        else:
            if methods is None:
                methods = ["tinystyler"]
            self._methods = DEFAULT_METHOD_REGISTRY.resolve(methods, device=device)
        if _mis_filter is not None:
            self._mis_filter = _mis_filter
        elif similarity_filter:
            self._mis_filter = MISFilter(device=device, **filter_kwargs)
        else:
            self._mis_filter: MISFilter | None = None

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
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = 51173,
        method_kwargs: Mapping[str, dict[str, Any]] | None = None,
        preprocess_kwargs: dict[str, Any] | None = None,
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
        max_new_tokens : int, optional
            Maximum number of tokens to generate per paraphrase.
            ``None`` lets each method choose its own default.
        temperature : float, optional
            Sampling temperature.  ``None`` lets each method choose
            its own default.
        top_p : float, optional
            Nucleus-sampling probability mass.  ``None`` lets each
            method choose its own default.
        seed : int
            Random seed for more reproducible output.  Seeds Python's
            ``random``, PyTorch (CPU + CUDA), and NumPy if available.
            Defaults to ``51173``.  Exact determinism is not guaranteed
            across different hardware or library versions.  Pass a
            different integer to get a new set of outputs, or ``None``
            to disable seeding.
        method_kwargs : mapping[str, dict], optional
            Per-method keyword arguments. Example:
            ``{"tinystyler": {"style_bank": [...]}}``.
        preprocess_kwargs : dict, optional
            Keyword arguments forwarded to
            :func:`~diversify_text._preprocess.preprocess`.  Example:
            ``{"split_on_punctuation": True}``.
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
        preprocess_kwargs = preprocess_kwargs or {}
        for method in self._methods:
            method.prepare()
        if self._mis_filter is not None:
            self._mis_filter.prepare()

        if seed is not None:
            import random
            import torch
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            try:
                import numpy as np
                np.random.seed(seed)
            except ImportError:
                pass
            logger.info("Using random seed: %d", seed)

        # --- process batches lazily ---
        n_candidates = (
            self._mis_filter.n_candidates
            if self._mis_filter is not None
            else 1
        )
        writer = OutputWriter(input_context, n_styles, out_path)
        writer.open()
        try:
            with tqdm(total=input_context.total, desc="Diversifying", unit="text") as pbar:
                while True:
                    batch_texts = list(islice(text_iter, batch_size))
                    if not batch_texts:
                        break

                    generation_texts, preprocess_context = preprocess(
                        batch_texts, **preprocess_kwargs
                    )

                    # Generate n_candidates sets of paraphrases.
                    all_candidates: list[list[list[str]]] = []
                    for _ in range(n_candidates):
                        candidate = self._diversify_batch(
                            batch_texts=generation_texts,
                            n_styles=n_styles,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            method_kwargs=method_kwargs,
                        )
                        candidate = postprocess(candidate, preprocess_context)
                        all_candidates.append(candidate)

                    if self._mis_filter is not None:
                        paraphrases_by_text = self._mis_filter.select_best(
                            batch_texts=batch_texts,
                            all_candidates=all_candidates,
                        )
                    else:
                        paraphrases_by_text = all_candidates[0]

                    writer.write_batch(batch_texts, paraphrases_by_text)
                    pbar.update(len(batch_texts))
        except Exception:
            writer.finish()
            raise

        return writer.finish()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
# Module-level convenience function (with per-model caching)
# ------------------------------------------------------------------


def diversify(
    texts: TextInput,
    *,
    device: str | None = None,
    methods: Sequence[str | DiversificationMethod] | None = None,
    similarity_filter: bool = False,
    **kwargs,
) -> DiversifyOutput:
    """One-shot convenience function: create a :class:`Diversifier` and run it.

    The generation method(s) and the MIS filter are cached independently
    between calls.  Generation methods are cached per (``device``,
    resolved method list), while the MIS filter is cached per ``device``.
    Switching ``similarity_filter`` on or off reuses the cached generation
    models, and changing methods reuses the cached MIS filter when
    possible.  Expensive components are only recreated when their
    respective cache keys change; changing filter thresholds (``min_score``,
    ``n_candidates``) updates the existing MIS filter instance rather than
    reloading it.

    Parameters
    ----------
    texts : str | list[str] | Iterable[str]
        Input text(s).
    device : str, optional
        Torch device.
    methods : sequence[str | DiversificationMethod], optional
        Method names and/or pre-built method instances.
    similarity_filter : bool
        When ``True``, score each paraphrase with the Mutual Implication
        Score model and select the best candidate above a minimum score.
    **kwargs
        Forwarded to :class:`Diversifier` (``min_score``,
        ``n_candidates``) and :meth:`Diversifier.diversify`
        (``n_styles``, ``text_column``, ``batch_size``,
        ``max_new_tokens``, ``temperature``, ``top_p``, ``seed``,
        ``method_kwargs``, ``preprocess_kwargs``,
        ``output_dir``, ``output_name``).

    Returns
    -------
    list[dict] | Path
        See :meth:`Diversifier.diversify`.

    Notes
    -----
    The internal cache is not thread-safe.  For multi-threaded
    applications, use :class:`Diversifier` directly.
    """
    # Separate filter kwargs from diversify() kwargs.
    filter_keys = {"min_score", "n_candidates"}
    filter_kwargs = {k: kwargs.pop(k) for k in filter_keys if k in kwargs}

    # Retrieve cached (or freshly resolved) components.
    cached_methods = _cache.get_methods(device, methods)
    mis_filter = _cache.get_mis_filter(device, **filter_kwargs) if similarity_filter else None

    # Build a Diversifier from the cached components.
    div = Diversifier(device=device, _methods=cached_methods, _mis_filter=mis_filter)

    return div.diversify(texts, **kwargs)
