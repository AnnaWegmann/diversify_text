"""Per-model caching for the :func:`~diversify_text.core.diversify` convenience function.

Keeps the generation method(s) and the MIS filter in independent
module-level caches so that toggling ``semantic_filter`` does not
reload the generation model, and switching methods does not reload the
MIS model.

Not thread-safe.  Intended for single-threaded use in scripts and
notebooks.  For multi-threaded applications, use :class:`Diversifier`
directly with your own instance management.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from diversify_text._utils import default_device
from diversify_text.filter.mis import MISFilter, _DEFAULT_MIN_SCORE, _DEFAULT_N_CANDIDATES
from diversify_text.method import DEFAULT_METHOD_REGISTRY, DiversificationMethod

_UNSET = object()  # sentinel that never equals a real cache key

_cached_methods: list[DiversificationMethod] | None = None
_cached_methods_key: object = _UNSET
_cached_mis_filter: MISFilter | None = None
_cached_mis_key: object = _UNSET


# kwargs that affect model construction and should invalidate the cache.
# Per-call kwargs (styles, prompts, n_style_examples, etc.) are excluded.
_CONSTRUCTOR_KWARGS = {"model", "device", "precision"}


def _methods_cache_key(
    device: str | None,
    methods: Sequence[str | DiversificationMethod] | None,
    method_kwargs: Mapping[str, dict[str, Any]] | None = None,
) -> tuple:
    """Build a hashable key for the generation method(s).

    Only includes constructor-level kwargs (e.g. ``model``) that affect
    which model is loaded.  Per-call kwargs (``styles``, ``prompts``,
    etc.) are excluded so changing them doesn't trigger a model reload.
    """
    methods_key: tuple = ()
    if methods is not None:
        methods_key = tuple(
            m if isinstance(m, str) else id(m) for m in methods
        )
    # Only include constructor kwargs in the cache key.
    mk_key: tuple = ()
    if method_kwargs:
        mk_key = tuple(
            (name, tuple(sorted(
                (k, v) for k, v in kw.items() if k in _CONSTRUCTOR_KWARGS
            )))
            for name, kw in sorted(method_kwargs.items())
        )
    return (device, methods_key, mk_key)


def _mis_cache_key(device: str | None) -> str | None:
    """Build a hashable key for the MIS filter (device only)."""
    return device


def get_methods(
    device: str | None,
    methods: Sequence[str | DiversificationMethod] | None,
    method_kwargs: Mapping[str, dict[str, Any]] | None = None,
) -> list[DiversificationMethod]:
    """Return cached generation methods, resolving only on config change.

    Per-method constructor kwargs (e.g. ``{"prompting": {"model": "..."}}``)
    are included in the cache key so that changing them triggers a fresh
    resolve.

    The current implementation treats the entire method list as a single
    cache key: any change (addition, removal, reordering) invalidates
    the whole cache and reloads all methods from scratch.  This is fine
    while only one method is used, but may cause unnecessary reloads
    when multiple methods are combined.  A future improvement could
    cache each method individually.
    """
    global _cached_methods, _cached_methods_key

    device = device or default_device()
    if methods is None:
        methods = ["tinystyler"]
    key = _methods_cache_key(device, methods, method_kwargs)
    if _cached_methods_key != key:
        # Merge per-method constructor kwargs into the resolve call.
        resolve_kwargs: dict[str, Any] = {"device": device}
        if method_kwargs:
            for mk in method_kwargs.values():
                resolve_kwargs.update(mk)
        _cached_methods = DEFAULT_METHOD_REGISTRY.resolve(
            methods,
            **resolve_kwargs,
        )
        _cached_methods_key = key
    assert _cached_methods is not None
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

    device = device or default_device()
    key = _mis_cache_key(device)
    if _cached_mis_key != key:
        _cached_mis_filter = MISFilter(device=device, **filter_kwargs)
        _cached_mis_key = key
    else:
        # Reset to defaults, then apply any overrides.
        assert _cached_mis_filter is not None
        _cached_mis_filter.min_score = filter_kwargs.get("min_score", _DEFAULT_MIN_SCORE)
        _cached_mis_filter.n_candidates = filter_kwargs.get("n_candidates", _DEFAULT_N_CANDIDATES)
    assert _cached_mis_filter is not None
    return _cached_mis_filter


def clear_cache() -> None:
    """Drop references to all cached models so their memory can be reclaimed when possible.

    This clears Python-level caches but does not guarantee immediate GPU/CPU
    memory release (e.g., allocator pools may retain reserved memory).
    """
    global _cached_methods, _cached_methods_key
    global _cached_mis_filter, _cached_mis_key

    _cached_methods = None
    _cached_methods_key = _UNSET
    _cached_mis_filter = None
    _cached_mis_key = _UNSET
