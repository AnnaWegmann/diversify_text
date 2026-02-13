"""Method registry and built-in registrations.
    based on: https://dev.to/dentedlogic/stop-writing-giant-if-else-chains-master-the-python-registry-pattern-ldm
"""

from __future__ import annotations

from typing import TypeAlias

from diversify.method.base import DiversificationMethod
from diversify.method.echo import EchoMethod
from diversify.method.tinystyler import TinyStylerMethod


MethodType: TypeAlias = type[DiversificationMethod]


class MethodRegistry:
    """Registry of named diversification method classes."""

    def __init__(self) -> None:
        self._store: dict[str, MethodType] = {}

    def register(self, name: str, method_cls: MethodType) -> None:
        if name in self._store:
            raise ValueError(f"Method '{name}' is already registered.")
        self._store[name] = method_cls

    def get(self, name: str) -> MethodType:
        if name not in self._store:
            available = ", ".join(sorted(self._store))
            raise KeyError(f"Unknown method '{name}'. Available: {available}")
        return self._store[name]

    def unregister(self, name: str) -> None:
        if name not in self._store:
            raise KeyError(f"Method '{name}' is not registered.")
        del self._store[name]

    def names(self) -> list[str]:
        return sorted(self._store)

    def __contains__(self, name: str) -> bool:
        return name in self._store


DEFAULT_METHOD_REGISTRY = MethodRegistry()
DEFAULT_METHOD_REGISTRY.register("echo", EchoMethod)
DEFAULT_METHOD_REGISTRY.register("tinystyler", TinyStylerMethod)
