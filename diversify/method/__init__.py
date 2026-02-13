"""Pluggable diversification approaches."""

from diversify.method.base import DiversificationApproach
from diversify.method.echo import EchoApproach
from diversify.method.registry import ApproachRegistry, DEFAULT_APPROACH_REGISTRY
from diversify.method.tinystyler import (
    StyleInput,
    TinyStyler,
    TinyStylerApproach,
    style_transfer,
)

__all__ = [
    "DiversificationApproach",
    "EchoApproach",
    "TinyStylerApproach",
    "StyleInput",
    "TinyStyler",
    "style_transfer",
    "ApproachRegistry",
    "DEFAULT_APPROACH_REGISTRY",
]
