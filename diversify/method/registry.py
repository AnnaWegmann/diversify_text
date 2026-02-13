"""Approach registry and built-in registrations."""

from __future__ import annotations

from typing import Any, Callable

from diversify.method.base import DiversificationApproach
from diversify.method.echo import EchoApproach
from diversify.method.tinystyler import TinyStylerApproach


ApproachFactory = Callable[..., DiversificationApproach]


class ApproachRegistry:
    """Registry of named diversification approach factories."""

    def __init__(self) -> None:
        self._factories: dict[str, ApproachFactory] = {}

    def register(self, name: str, factory: ApproachFactory) -> None:
        self._factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> DiversificationApproach:
        if name not in self._factories:
            available = ", ".join(sorted(self._factories))
            raise ValueError(f"Unknown approach '{name}'. Available: {available}")
        return self._factories[name](**kwargs)


DEFAULT_APPROACH_REGISTRY = ApproachRegistry()
DEFAULT_APPROACH_REGISTRY.register("echo", lambda **kwargs: EchoApproach())
DEFAULT_APPROACH_REGISTRY.register(
    "tinystyler",
    lambda **kwargs: TinyStylerApproach(device=kwargs.get("device")),
)
