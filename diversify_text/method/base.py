"""Abstract base class for diversification methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DiversificationMethod(ABC):
    """Interface for pluggable diversification methods."""

    name = "base"

    def prepare(self) -> None:
        """Load any resources (models, tokenizers) needed before generation.

        Called once before the progress bar starts so that loading messages
        appear before generation begins.  No-op by default.
        """

    @abstractmethod
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
        """Return paraphrases per input text.

        *style_dict* maps each target style name to its example texts
        (as built by :func:`~diversify_text.styles.resolve_style_dict`).
        For each input text, return one generated string per style, in
        *style_dict* order — output shape ``len(texts)`` x
        ``len(style_dict)``.
        """
