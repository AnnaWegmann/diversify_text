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
        *,
        n: int,
        max_new_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        **kwargs: Any,
    ) -> list[list[str]]:
        return [[text for _ in range(n)] for text in texts]
