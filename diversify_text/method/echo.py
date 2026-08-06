"""Deterministic fallback diversification method."""

from __future__ import annotations

from typing import Any

from diversify_text.method.base import DiversificationMethod


class EchoMethod(DiversificationMethod):
    """Echoes the input text for every requested style."""

    name = "echo"

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
        return [[text for _ in style_dict] for text in texts]
