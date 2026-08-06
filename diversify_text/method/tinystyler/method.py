"""TinyStyler-backed diversification method."""

from __future__ import annotations

import logging
from typing import Any

from diversify_text.method.base import DiversificationMethod
from diversify_text.method.tinystyler.model import TinyStyler, get_tinystyler

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
        """Fetch the shared model on first use."""
        if self._model is None:
            self._model = get_tinystyler(self.device)
        return self._model

    def generate(
        self,
        texts: list[str],
        style_dict: dict[str, list[str]],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs: Any,
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
            Generation parameters; ``None`` uses defaults.

        Returns
        -------
        list[list[str]]
            For each input text, one generated string per style, in
            *style_dict* order.
        """
        model = self._ensure_model()

        # Apply TinyStyler-specific defaults for parameters not set by
        # the caller.
        temperature = temperature if temperature is not None else _DEFAULT_TEMPERATURE
        top_p = top_p if top_p is not None else _DEFAULT_TOP_P

        # An explicit max_new_tokens is used as-is; otherwise scale with
        # the longest input, capped between _MAX_NEW_TOKENS_FLOOR and
        # _MAX_NEW_TOKENS_CAP.
        if max_new_tokens is None:
            input_token_counts = [
                len(ids)
                for ids in model._tokenizer(texts, truncation=True)["input_ids"]
            ]
            max_new_tokens = max(
                _MAX_NEW_TOKENS_FLOOR,
                min(
                    int(max(input_token_counts) * _MAX_NEW_TOKENS_FACTOR),
                    _MAX_NEW_TOKENS_CAP,
                ),
            )

        paraphrases_per_text = [[] for _ in texts]

        for style_examples in style_dict.values():
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
