"""TinyStyler-backed diversification method."""

from __future__ import annotations

import logging
from typing import Any

from diversify_text.method.base import DiversificationMethod
from diversify_text.method.tinystyler.model import TinyStyler
from diversify_text.method.tinystyler.styles import DEFAULT_STYLE_BANK, DEFAULT_STYLES

logger = logging.getLogger(__name__)

_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_TOP_P = 0.9
_MAX_NEW_TOKENS_FACTOR = 2.0
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

        Parameters
        ----------
        style_bank : dict | list | None
            A custom style bank.  ``None`` falls back to
            :data:`DEFAULT_STYLE_BANK`.
        styles : list[str] | None
            When provided, select only these keys from the bank (which must
            be dict-shaped).  Order is preserved.

        Returns
        -------
        list[list[str]]
            One list of example strings per selected style.
        """
        # --- obtain a dict bank when possible ---
        if style_bank is None:
            bank_dict: dict[str, list[str]] | None = DEFAULT_STYLE_BANK
        elif isinstance(style_bank, dict):
            bank_dict = style_bank
        else:
            bank_dict = None  # list-format, can't filter by key

        # --- filter by key names ---
        if styles is not None:
            if bank_dict is None:
                raise TypeError(
                    "Cannot use 'styles' key selection with a list-format "
                    "style bank. Pass a dict-format bank or use the default."
                )
            unknown = set(styles) - set(bank_dict.keys())
            if unknown:
                raise ValueError(
                    f"Unknown style key(s): {sorted(unknown)}. "
                    f"Available: {sorted(bank_dict.keys())}"
                )
            return [bank_dict[k] for k in styles]

        # --- no key filter: return everything ---
        if bank_dict is not None:
            return list(bank_dict.values())

        # legacy list-format normalisation
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
                    "Each style group must be either a string or list[str]."
                )
        if not normalized:
            raise ValueError("style_bank cannot be empty.")
        return normalized

    def generate(
        self,
        texts: list[str],
        *,
        n_styles: int,
        max_new_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        **kwargs: Any,
    ) -> list[list[str]]:
        model = self._ensure_model()

        # Apply TinyStyler-specific defaults for parameters not set by
        # the caller.
        temperature = temperature if temperature is not None else _DEFAULT_TEMPERATURE
        top_p = top_p if top_p is not None else _DEFAULT_TOP_P

        # Cap max_new_tokens at _MAX_NEW_TOKENS_FACTOR the longest input or _MAX_NEW_TOKENS_CAP, whichever
        # is smaller.  An explicit caller value is used as-is.
        input_token_counts = [
            len(ids)
            for ids in model._tokenizer(texts, truncation=True)["input_ids"]
        ]
        dynamic_cap = min(
            int(max(input_token_counts) * _MAX_NEW_TOKENS_FACTOR),
            _MAX_NEW_TOKENS_CAP,
        )
        max_new_tokens = max_new_tokens if max_new_tokens is not None else dynamic_cap

        styles_arg = kwargs.get("styles")
        if styles_arg is None and kwargs.get("style_bank") is None:
            styles_arg = DEFAULT_STYLES[:n_styles]
        style_bank = self._resolve_styles(
            kwargs.get("style_bank"),
            styles_arg,
        )
        # When explicit style keys are given, they determine the count.
        effective_n = len(styles_arg) if styles_arg is not None else n_styles
        if effective_n > len(style_bank):
            logger.warning(
                "n_styles=%d exceeds the number of style bank entries (%d). "
                "Styles will wrap around, producing repeated style patterns. "
                "Consider adding more entries to the style bank.",
                effective_n, len(style_bank),
            )
        if styles_arg is not None:
            logger.info("Using styles: %s", ", ".join(styles_arg))
        else:
            logger.info("Using %d style(s) from style bank.", effective_n)
        paraphrases_per_text = [[] for _ in texts]

        for i in range(effective_n):
            style_examples = style_bank[i % len(style_bank)]
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
