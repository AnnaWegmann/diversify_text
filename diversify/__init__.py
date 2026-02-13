"""diversify -- generate stylistic paraphrases of texts."""

from diversify.method import (
    ApproachRegistry,
    DiversificationApproach,
    EchoApproach,
    TinyStylerApproach,
)
from diversify.core import (
    Diversifier,
    diversify,
)

__all__ = [
    "DiversificationApproach",
    "ApproachRegistry",
    "EchoApproach",
    "TinyStylerApproach",
    "Diversifier",
    "diversify",
]
