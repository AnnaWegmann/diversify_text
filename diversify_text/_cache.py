"""Per-model caching for the :func:`~diversify_text.core.diversify` convenience function.

Keeps the generation method(s) and the MIS filter in independent
module-level caches so that toggling ``similarity_filter`` does not
reload the generation model, and switching methods does not reload the
MIS model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from diversify_text.filter.mis import MISFilter
from diversify_text.method import DEFAULT_METHOD_REGISTRY, DiversificationMethod

_cached_methods: list[DiversificationMethod] | None = None
_cached_methods_key: tuple | None = None
_cached_mis_filter: MISFilter | None = None
_cached_mis_key: str | None = None


def _methods_cache_key(
    device: str | None,
    methods: Sequence[str | DiversificationMethod] | None,
) -> tuple:
    """Build a hashable key for the generation method(s)."""
    methods_key: tuple = ()
    if methods is not None:
        methods_key = tuple(
            m if isinstance(m, str) else id(m) for m in methods
        )
    return (device, methods_key)


def _mis_cache_key(device: str | None) -> str | None:
    """Build a hashable key for the MIS filter (device only)."""
    return device


def get_methods(
    device: str | None,
    methods: Sequence[str | DiversificationMethod] | None,
) -> list[DiversificationMethod]:
    """Return cached generation methods, resolving only on config change."""
    global _cached_methods, _cached_methods_key

    key = _methods_cache_key(device, methods)
    if _cached_methods_key != key:
        _cached_methods = DEFAULT_METHOD_REGISTRY.resolve(
            methods if methods is not None else ["tinystyler"],
            device=device,
        )
        _cached_methods_key = key
    return _cached_methods


def get_mis_filter(
    device: str | None,
    **filter_kwargs: Any,
) -> MISFilter:
    """Return cached MIS filter, reloading only when *device* changes.

    Threshold settings (``min_score``, ``n_candidates``) are updated on
    the existing instance without reloading the model.
    """
    global _cached_mis_filter, _cached_mis_key

    key = _mis_cache_key(device)
    if _cached_mis_key != key:
        _cached_mis_filter = MISFilter(device=device, **filter_kwargs)
        _cached_mis_key = key
    else:
        # Update settings without reloading.
        if "min_score" in filter_kwargs:
            _cached_mis_filter.min_score = filter_kwargs["min_score"]
        if "n_candidates" in filter_kwargs:
            _cached_mis_filter.n_candidates = filter_kwargs["n_candidates"]
    return _cached_mis_filter


def clear_cache() -> None:
    """Free all cached models and release their memory."""
    global _cached_methods, _cached_methods_key
    global _cached_mis_filter, _cached_mis_key

    _cached_methods = None
    _cached_methods_key = None
    _cached_mis_filter = None
    _cached_mis_key = None
