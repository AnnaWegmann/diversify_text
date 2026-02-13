"""Deterministic fallback diversification method."""

from __future__ import annotations

from typing import Any

from diversify.method.base import DiversificationMethod


class EchoMethod(DiversificationMethod):
    """Echoes the input text for every requested style."""

    name = "echo"

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
        return [[text for _ in range(n_styles)] for text in texts]
