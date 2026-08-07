"""
Core module for text diversification via stylistic paraphrasing.

Provides the :class:`Diversifier` class and a convenience :func:`diversify`
function that produce multiple stylistically varied paraphrases for each
input text.
"""

from __future__ import annotations

from collections.abc import Mapping
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
from diversify_text.styles import DEFAULT_STYLE_BANK, resolve_style_dict

logger = logging.getLogger(__name__)

_DEFAULT_SEED = 51173
_DEFAULT_METHOD = "tinystyler"
_SENTINEL = object()
_default_seed_applied: bool = False


class Diversifier:
    """Generate stylistic paraphrases using a pluggable method.

    The method is the model or algorithm doing the style transfer in
    the background.

    Parameters
    ----------
    device : str, optional
        Torch device (``"cuda"``, ``"cpu"``, ``"mps"``, ...).
    method : str | DiversificationMethod, optional
        A built-in method name (default ``"tinystyler"``) or a
        pre-built method instance.

    Example
    -------
    >>> div = Diversifier(method="tinystyler")
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
        method: str | DiversificationMethod | None = None,
        semantic_filter: bool = False,
        _mis_filter: MISFilter | None = None,
        **filter_kwargs: Any,
    ) -> None:
        self.device = device
        if method is None:
            method = _DEFAULT_METHOD
        self._method = DEFAULT_METHOD_REGISTRY.resolve(method, device=device)
        if _mis_filter is not None:
            self._mis_filter = _mis_filter
        elif semantic_filter:
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
        n: int | None = None,
        styles: list[str | int] | None = None,
        style_texts: list[str] | list[list[str]] | dict[str, list[str]] | None = None,
        repeats: int = 1,
        text_column: str = "text",
        batch_size: int = 32,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None | object = _SENTINEL,
        method_kwargs: Mapping[str, Any] | None = None,
        preprocess_kwargs: dict[str, Any] | None = None,
        output_dir: str | Path | None = None,
        output_name: str | None = None,
    ) -> DiversifyOutput:
        """Produce one stylistic paraphrase per target style for each input text.

        Parameters
        ----------
        texts : str | list[str] | Iterable[str]
            A single text, a list of texts, a generator/iterable of texts,
            or a path to a ``.csv``, ``.tsv``, or ``.txt`` file.
        n : int or None
            Number of distinct styles to draw from the default styles
            (and therefore paraphrases per text) when neither *styles*
            nor *style_texts* is given.  Defaults to ``5``.  Cannot
            be combined with *styles* or *style_texts* — the number
            of styles already determines the number of paraphrases.
        styles : list of str or int, optional
            Selection from the built-in style bank, by name
            (``"recipe"``) and/or 0-based index (``7``).
        style_texts : list[str] | list[list[str]] | dict, optional
            Your own target styles, defined by example texts: a flat
            list is one style, a list of lists is several styles, a
            dict maps style names to example texts.  Can be combined
            with *styles*.
        repeats : int
            How many paraphrases to generate per style (default 1).
            The output interleaves the styles: style A, style B,
            style A, style B.  With the semantic filter on, generation
            cost is styles x repeats x ``n_candidates``.
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
        seed : int or None, optional
            Random seed for reproducible output.  Seeds Python's
            ``random``, PyTorch (CPU + CUDA), and NumPy if available.
            When omitted, the default seed (``51173``) is applied on the
            first call only and skipped on subsequent calls.  Pass an
            explicit integer to always (re-)seed.  Pass ``None`` to
            disable seeding entirely.
        method_kwargs : mapping[str, Any], optional
            Method-specific keyword arguments. Example:
            ``{"prompt": "humanize_transfer"}`` for prompting.
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

                {"original": str,
                 "paraphrases": [{"style": str, "text": str}, ...]}

            Otherwise, returns the ``Path`` to the output file(s).
        """
        # Resolve the target styles into one style dict.  Explicit
        # styles determine the number of paraphrases; n only selects
        # from the default styles when nothing else is given.
        if styles is not None or style_texts is not None:
            if n is not None:
                raise ValueError(
                    "n cannot be combined with styles or style_texts "
                    "— the number of styles already determines the "
                    "number of paraphrases."
                )
            style_dict = resolve_style_dict(styles, style_texts)
        else:
            if n is None:
                n = self._DEFAULT_N
            if n < 1:
                raise ValueError("n must be >= 1.")
            if n > len(DEFAULT_STYLE_BANK):
                raise ValueError(
                    f"n={n} exceeds the number of available styles "
                    f"({len(DEFAULT_STYLE_BANK)})."
                )
            style_dict = resolve_style_dict(styles=list(DEFAULT_STYLE_BANK)[:n])
        if repeats < 1:
            raise ValueError("repeats must be >= 1.")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")

        # --- resolve input & output ---
        text_iter, input_context = resolve_input(texts, text_column)
        out_path = resolve_output_path(input_context, output_dir, output_name)

        # --- prepare the method (model loading etc.) ---
        method_kwargs = method_kwargs or {}
        preprocess_kwargs = preprocess_kwargs or {}
        self._method.prepare()
        if self._mis_filter is not None:
            self._mis_filter.prepare()

        global _default_seed_applied
        if seed is _SENTINEL:
            if not _default_seed_applied:
                self._apply_seed(_DEFAULT_SEED)
                _default_seed_applied = True
        elif seed is not None:
            self._apply_seed(seed)

        # --- process batches lazily ---
        n_candidates = (
            self._mis_filter.n_candidates
            if self._mis_filter is not None
            else 1
        )
        writer = OutputWriter(input_context, len(style_dict) * repeats, out_path)
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

                    # Generate n_candidates sets of paraphrases.  Each
                    # set holds one full style round per repeat, so the
                    # styles interleave: style A, style B, style A, ...
                    all_candidates: list[list[list[str]]] = []
                    for _ in range(n_candidates):
                        candidate = [[] for _ in generation_texts]
                        for _ in range(repeats):
                            style_round = self._diversify_batch(
                                batch_texts=generation_texts,
                                style_dict=style_dict,
                                max_new_tokens=max_new_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                method_kwargs=method_kwargs,
                            )
                            for row, extra in zip(candidate, style_round):
                                row.extend(extra)
                        candidate = postprocess(candidate, preprocess_context)
                        all_candidates.append(candidate)

                    if self._mis_filter is not None:
                        paraphrases_by_text = self._mis_filter.select_best(
                            batch_texts=batch_texts,
                            all_candidates=all_candidates,
                        )
                    else:
                        paraphrases_by_text = all_candidates[0]

                    # Label each paraphrase with the style that produced
                    # it (the style names cycle once per repeat).
                    slot_styles = list(style_dict) * repeats
                    labeled_by_text = [
                        [
                            {"style": style_name, "text": paraphrase}
                            for style_name, paraphrase in zip(slot_styles, row)
                        ]
                        for row in paraphrases_by_text
                    ]
                    writer.write_batch(batch_texts, labeled_by_text)
                    pbar.update(len(batch_texts))
        except Exception:
            writer.finish()
            raise

        return writer.finish()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _DEFAULT_N = 5

    @staticmethod
    def _apply_seed(seed: int) -> None:
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

    def _diversify_batch(
        self,
        *,
        batch_texts: list[str],
        style_dict: dict[str, list[str]],
        max_new_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        method_kwargs: Mapping[str, Any],
    ) -> list[list[str]]:
        paraphrases_by_text = self._method.generate(
            batch_texts,
            style_dict,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **method_kwargs,
        )
        self._validate_paraphrases(paraphrases_by_text, batch_texts)
        return paraphrases_by_text

    @staticmethod
    def _validate_paraphrases(
        incoming: list[list[str]],
        source_texts: list[str],
    ) -> None:
        """Validate the shape of a method's output."""
        if len(incoming) != len(source_texts):
            raise ValueError("Method returned invalid batch size.")
        generated_styles: int | None = None
        for group in incoming:
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


# ------------------------------------------------------------------
# Module-level convenience function (with per-model caching)
# ------------------------------------------------------------------


def diversify(
    texts: TextInput,
    *,
    device: str | None = None,
    method: str | DiversificationMethod | None = None,
    semantic_filter: bool = False,
    **kwargs,
) -> DiversifyOutput:
    """One-shot convenience function: create a :class:`Diversifier` and run it.

    Loaded models are cached between calls: the generation model per
    configuration (``model``, ``device``, ``precision``) and the MIS
    filter per ``device``, independently of each other.  Switching
    ``semantic_filter`` on or off reuses the cached generation model,
    changing the method reuses the cached MIS filter, and per-call
    options never trigger a model reload; changing filter thresholds
    (``min_score``, ``n_candidates``) updates the existing MIS filter
    instance rather than reloading it.

    Parameters
    ----------
    texts : str | list[str] | Iterable[str]
        Input text(s).
    device : str, optional
        Torch device.
    method : str | DiversificationMethod, optional
        A built-in method name (default ``"tinystyler"``) or a
        pre-built method instance.
    semantic_filter : bool
        When ``True``, score each paraphrase with the Mutual Implication
        Score model and select the best candidate above a minimum score.
    **kwargs
        Forwarded to :class:`Diversifier` (``min_score``,
        ``n_candidates``) and :meth:`Diversifier.diversify`
        (``n``, ``styles``, ``style_texts``, ``repeats``,
        ``text_column``, ``batch_size``, ``max_new_tokens``,
        ``temperature``, ``top_p``, ``seed``, ``method_kwargs``,
        ``preprocess_kwargs``, ``output_dir``, ``output_name``).

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

    # Resolve the method (same default as Diversifier).  No model is
    # loaded and nothing is cached here — the method object just stores
    # its configuration.  The model is loaded on first use, when
    # div.diversify() calls the method's prepare(): that fetches the
    # model through the method's @model_cache loader (see _cache.py),
    # which is where reuse across calls happens.  Constructor arguments
    # in method_kwargs (e.g. ``model``) are applied here; per-call
    # arguments are applied at generation time.
    resolve_kwargs: dict[str, Any] = {"device": device}
    if kwargs.get("method_kwargs"):
        resolve_kwargs.update(kwargs["method_kwargs"])
    resolved_method = DEFAULT_METHOD_REGISTRY.resolve(
        method if method is not None else _DEFAULT_METHOD, **resolve_kwargs
    )

    mis_filter = _cache.get_cached_mis_filter(device, **filter_kwargs) if semantic_filter else None

    # The filter has no public object parameter, hence the private one.
    div = Diversifier(device=device, method=resolved_method, _mis_filter=mis_filter)

    return div.diversify(texts, **kwargs)
