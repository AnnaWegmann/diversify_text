"""Method registry and built-in registrations."""

from __future__ import annotations

from typing import Any, Callable

from diversify.method.base import DiversificationMethod
from diversify.method.echo import EchoMethod
from diversify.method.tinystyler import TinyStylerMethod


MethodFactory = Callable[..., DiversificationMethod]


class MethodRegistry:
    """Registry of named diversification method factories."""

    def __init__(self) -> None:
        self._factories: dict[str, MethodFactory] = {}

    def register(self, name: str, factory: MethodFactory) -> None:
        self._factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> DiversificationMethod:
        if name not in self._factories:
            available = ", ".join(sorted(self._factories))
            raise ValueError(f"Unknown method '{name}'. Available: {available}")
        return self._factories[name](**kwargs)


DEFAULT_METHOD_REGISTRY = MethodRegistry()
DEFAULT_METHOD_REGISTRY.register("echo", lambda **kwargs: EchoMethod())
DEFAULT_METHOD_REGISTRY.register(
    "tinystyler",
    lambda **kwargs: TinyStylerMethod(device=kwargs.get("device")),
)
