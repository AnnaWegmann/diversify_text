"""TinyStyler-backed diversification method."""

from __future__ import annotations

import logging
from typing import Any

from diversify_text.method.base import DiversificationMethod
from diversify_text.method.tinystyler.model import TinyStyler
from diversify_text.styles import DEFAULT_STYLES, resolve_style_sets

logger = logging.getLogger(__name__)

_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_TOP_P = 0.9
_MAX_NEW_TOKENS_FACTOR = 2.0
_MAX_NEW_TOKENS_FLOOR = 10
_MAX_NEW_TOKENS_CAP = 256


class TinyStylerMethod(DiversificationMethod):
    """Diversification method backed by TinyStyler."""

    name = "tinystyler"

    def __init__(self, device: str | None = None) -> None:
        self.device = device
        self._model: TinyStyler | None = None

    def prepare(self) -> None:
        self._ensure_model()

    def _ensure_model(self) -> TinyStyler:
        if self._model is None:
            self._model = TinyStyler(device=self.device)
        return self._model

    @staticmethod
    def _resolve_styles(
        style_bank: Any = None,
        styles: list[str] | None = None,
    ) -> list[list[str]]:
        """Resolve *style_bank* and optional *styles* key filter.

        Delegates dict-based resolution to the shared
        :func:`~diversify_text.styles.resolve_style_sets`.
        Also supports legacy list-format style banks.

        Returns
        -------
        list[list[str]]
            One list of example strings per selected style.
        """
        # Dict or default → use the shared resolver.
        if style_bank is None or isinstance(style_bank, dict):
            return list(resolve_style_sets(style_bank, styles).values())

        # Legacy list-format: can't filter by key.
        if styles is not None:
            raise TypeError(
                "Cannot use 'styles' key selection with a list-format "
                "style bank. Pass a dict-format bank or use the default."
            )
        if not isinstance(style_bank, list):
            raise TypeError("style_bank must be a list or dict of style example groups.")
        normalized: list[list[str]] = []
        for group in style_bank:
            if isinstance(group, str):
                normalized.append([group])
            elif isinstance(group, list) and all(isinstance(x, str) for x in group):
                normalized.append(group)
            else:
                raise TypeError(
                    "Each style set must be either a string or list[str]."
                )
        if not normalized:
            raise ValueError("style_bank cannot be empty.")
        return normalized

    def generate(
        self,
        texts: list[str],
        *,
        n: int,
        max_new_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        **kwargs: Any,
    ) -> list[list[str]]:
        # Resolve styles and check n before any model work, so bad
        # input fails fast without loading a model.
        styles_arg = kwargs.get("styles")
        if styles_arg is None and kwargs.get("style_bank") is None:
            styles_arg = DEFAULT_STYLES[:n]
        style_bank = self._resolve_styles(
            kwargs.get("style_bank"),
            styles_arg,
        )
        if n > len(style_bank):
            raise ValueError(
                f"n={n} exceeds the number of available styles "
                f"({len(style_bank)})."
            )
        if styles_arg is not None:
            logger.info("Using styles: %s", ", ".join(styles_arg))
        else:
            logger.info("Using %d style(s) from style bank.", n)

        return self._generate_from_style_dict_list(
            texts,
            style_bank[:n],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def _generate_from_style_dict(
        self,
        texts: list[str],
        style_dict: dict[str, list[str]],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> list[list[str]]:
        """Generate one paraphrase per style in *style_dict* for each text.

        Parameters
        ----------
        texts : list[str]
            Input texts to paraphrase.
        style_dict : dict[str, list[str]]
            Maps each target style name to its example texts (as built
            by :func:`~diversify_text.styles.resolve_style_dict`).
        max_new_tokens, temperature, top_p
            As in :meth:`generate`; ``None`` uses defaults.

        Returns
        -------
        list[list[str]]
            For each input text, one generated string per style, in
            *style_dict* order.

        This becomes the ``generate()`` interface in #18; nothing calls
        it yet.
        """
        return self._generate_from_style_dict_list(
            texts,
            list(style_dict.values()),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def _generate_from_style_dict_list(
        self,
        texts: list[str],
        style_sets: list[list[str]],
        *,
        max_new_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
    ) -> list[list[str]]:
        """Shared generation loop: one transfer per style example set."""
        model = self._ensure_model()

        # Apply TinyStyler-specific defaults for parameters not set by
        # the caller.
        temperature = temperature if temperature is not None else _DEFAULT_TEMPERATURE
        top_p = top_p if top_p is not None else _DEFAULT_TOP_P

        # Cap max_new_tokens between _MAX_NEW_TOKENS_FLOOR and
        # _MAX_NEW_TOKENS_CAP, scaling with the longest input.
        # An explicit caller value is used as-is.
        input_token_counts = [
            len(ids)
            for ids in model._tokenizer(texts, truncation=True)["input_ids"]
        ]
        dynamic_cap = max(
            _MAX_NEW_TOKENS_FLOOR,
            min(
                int(max(input_token_counts) * _MAX_NEW_TOKENS_FACTOR),
                _MAX_NEW_TOKENS_CAP,
            ),
        )
        max_new_tokens = max_new_tokens if max_new_tokens is not None else dynamic_cap

        paraphrases_per_text = [[] for _ in texts]

        for style_examples in style_sets:
            batch = model.transfer(
                texts,
                style_examples,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            for row_idx, generated in enumerate(batch):
                paraphrases_per_text[row_idx].append(generated)
        return paraphrases_per_text
