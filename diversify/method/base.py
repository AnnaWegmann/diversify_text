"""Abstract base class for diversification approaches."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DiversificationApproach(ABC):
    """Interface for pluggable diversification approaches."""

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
