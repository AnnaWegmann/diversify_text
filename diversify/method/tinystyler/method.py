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
    def _normalize_style_bank(style_bank: Any) -> list[list[str]]:
        if style_bank is None:
            return DEFAULT_STYLE_BANK
        if not isinstance(style_bank, list):
            raise TypeError("style_bank must be a list of style example groups.")
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
        style_bank = self._normalize_style_bank(kwargs.get("style_bank"))
        if n_styles > len(style_bank):
            logger.warning(
                "n_styles=%d exceeds the number of style bank entries (%d). "
                "Styles will wrap around, producing repeated style patterns. "
                "Consider adding more entries to the style bank.",
                n_styles, len(style_bank),
            )
        paraphrases_per_text = [[] for _ in texts]

        for i in range(n_styles):
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
