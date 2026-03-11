"""Pluggable diversification methods."""

from diversify_text.method.base import DiversificationMethod
from diversify_text.method.registry import (
    DEFAULT_METHOD_REGISTRY,
    MethodRegistry,
)

__all__ = [
    "DiversificationMethod",
    "MethodRegistry",
    "DEFAULT_METHOD_REGISTRY",
]
