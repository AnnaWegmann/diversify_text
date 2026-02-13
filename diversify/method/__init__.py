"""Pluggable diversification methods."""

from diversify.method.base import DiversificationMethod
from diversify.method.registry import (
    DEFAULT_METHOD_REGISTRY,
    MethodRegistry,
)

__all__ = [
    "DiversificationMethod",
    "MethodRegistry",
    "DEFAULT_METHOD_REGISTRY",
]
