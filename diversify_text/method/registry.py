"""Method registry and built-in registrations.
    based on: https://dev.to/dentedlogic/stop-writing-giant-if-else-chains-master-the-python-registry-pattern-ldm
"""

from __future__ import annotations

import inspect
from typing import Any, TypeAlias

from diversify_text.method.base import DiversificationMethod
from diversify_text.method.echo import EchoMethod
from diversify_text.method.prompting import PromptingMethod
from diversify_text.method.tinystyler import TinyStylerMethod
from diversify_text.method.zero_shot import ZeroShotMethod


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

    def resolve(
        self,
        method: str | DiversificationMethod,
        **kwargs: Any,
    ) -> DiversificationMethod:
        """Resolve a method name or instance into a ready instance.

        A string is looked up in the registry and instantiated.  A
        pre-built :class:`DiversificationMethod` instance is passed
        through as-is.  Extra *kwargs* (e.g. ``device``) are forwarded
        to the constructor only if it accepts them.

        Parameters
        ----------
        method : str | DiversificationMethod
            Method name or pre-built instance.
        **kwargs
            Keyword arguments forwarded to the method constructor
            (only those accepted by the constructor signature).

        Returns
        -------
        DiversificationMethod

        Raises
        ------
        KeyError
            If a string name is not registered.
        TypeError
            If *method* is neither ``str`` nor
            :class:`DiversificationMethod`.
        """
        if isinstance(method, DiversificationMethod):
            return method
        if isinstance(method, str):
            method_class = self.get(method)
            init_kwargs = _build_init_kwargs(method_class, kwargs)
            return method_class(**init_kwargs)
        raise TypeError(
            "method must be str or DiversificationMethod instance."
        )

    def unregister(self, name: str) -> None:
        if name not in self._store:
            raise KeyError(f"Method '{name}' is not registered.")
        del self._store[name]

    def names(self) -> list[str]:
        return sorted(self._store)

    def __contains__(self, name: str) -> bool:
        return name in self._store


def _build_init_kwargs(
    method_cls: type[DiversificationMethod],
    available: dict[str, Any],
) -> dict[str, Any]:
    """Return only the *available* kwargs that *method_cls* accepts."""
    sig = inspect.signature(method_cls)
    return {k: v for k, v in available.items() if k in sig.parameters}


DEFAULT_METHOD_REGISTRY = MethodRegistry()
DEFAULT_METHOD_REGISTRY.register("echo", EchoMethod)
DEFAULT_METHOD_REGISTRY.register("prompting", PromptingMethod)
DEFAULT_METHOD_REGISTRY.register("tinystyler", TinyStylerMethod)
DEFAULT_METHOD_REGISTRY.register("zero_shot", ZeroShotMethod)
