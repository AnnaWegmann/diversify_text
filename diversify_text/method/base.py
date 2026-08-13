"""Abstract base class for diversification methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from diversify_text.styles import (
    DEFAULT_STYLE_BANK,
    SURFACE_STYLE_BANK,
    UNUSUAL_STYLE_BANK,
)


class DiversificationMethod(ABC):
    """Interface for pluggable diversification methods."""

    name = "base"

    #: The style bank this method selects from: style name → the texts
    #: that define the style.  ``styles`` names/indices and the
    #: ``n``-default pool are resolved against the active method's
    #: bank.  Methods may override it with their own bank.
    style_bank: dict[str, list[str]] = DEFAULT_STYLE_BANK

    #: Extra styles selectable by *name* only: never part of the ``n``
    #: pool and not addressable by index.  Methods that override
    #: ``style_bank`` should usually override this too (typically with
    #: ``{}``, unless their bank pairs with the default unusual set).
    unusual_style_bank: dict[str, list[str]] = UNUSUAL_STYLE_BANK

    #: Surface-level rewrites (all caps, passive voice, ...), also
    #: selectable by name only.  Same override rule as
    #: ``unusual_style_bank``.
    surface_style_bank: dict[str, list[str]] = SURFACE_STYLE_BANK

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
