"""Abstract base class for diversification methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DiversificationMethod(ABC):
    """Interface for pluggable diversification methods."""

    name = "base"

    @abstractmethod
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
        """Return paraphrases per input text.

        Output shape must be ``len(texts)`` x ``n_styles``.
        """
