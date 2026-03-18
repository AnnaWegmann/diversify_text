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

    def reset(self) -> None:
        """Reset any per-run state.

        Called before each :meth:`~Diversifier.diversify` batch loop so
        that methods can clear accumulated state (e.g. diversity
        selectors).  No-op by default.
        """

    @abstractmethod
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
        """Return paraphrases per input text.

        Output shape must be ``len(texts)`` x ``n_styles``.
        """
