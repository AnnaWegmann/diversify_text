"""Pluggable diversification methods."""

from diversify.method.base import DiversificationMethod
from diversify.method.echo import EchoMethod
from diversify.method.registry import (
    DEFAULT_METHOD_REGISTRY,
    MethodRegistry,
)
from diversify.method.tinystyler import (
    StyleInput,
    TinyStyler,
    TinyStylerMethod,
    style_transfer,
)

__all__ = [
    "DiversificationMethod",
    "EchoMethod",
    "TinyStylerMethod",
    "StyleInput",
    "TinyStyler",
    "style_transfer",
    "MethodRegistry",
    "DEFAULT_METHOD_REGISTRY",
]
