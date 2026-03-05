"""TinyStyler-backed diversification method."""

from __future__ import annotations

import logging
from typing import Any

from diversify.method.base import DiversificationMethod
from diversify.method.tinystyler.model import TinyStyler
from diversify.method.tinystyler.styles import DEFAULT_STYLE_BANK

logger = logging.getLogger(__name__)


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
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs: Any,
    ) -> list[list[str]]:
        model = self._ensure_model()
        styles_arg = kwargs.get("styles")
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
