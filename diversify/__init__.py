"""diversify -- generate stylistic paraphrases of texts."""

from diversify.method import (
    MethodRegistry,
    DiversificationMethod,
    EchoMethod,
    TinyStylerMethod,
)
from diversify.core import (
    Diversifier,
    diversify,
)

__all__ = [
    "DiversificationMethod",
    "MethodRegistry",
    "EchoMethod",
    "TinyStylerMethod",
    "Diversifier",
    "diversify",
]
