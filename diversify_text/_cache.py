"""Model caching.

Loaded models are cached, method instances are not: method objects are
cheap to create, while model loading is expensive.  Every model module
wraps its loader in :func:`model_cache`, which keeps the most recently
loaded model per family (switching configuration drops the previous
model automatically) and registers the loader so :func:`clear_cache`
can drop everything.  The cached loaders live next to their model
classes: the causal LM engine in :mod:`diversify_text.method.llm`, the
TinyStyler model in :mod:`diversify_text.method.tinystyler.model`, and
the MIS filter below.

Not thread-safe.  Intended for single-threaded use in scripts and
notebooks.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, TypeVar

from diversify_text._utils import default_device
from diversify_text.filter.mis import MISFilter, _DEFAULT_MIN_SCORE, _DEFAULT_N_CANDIDATES

_Loader = TypeVar("_Loader", bound=Callable)

#: All cached loaders, registered by :func:`model_cache`.
_MODEL_CACHES: list[Any] = []


def model_cache(loader: _Loader) -> _Loader:
    """Cache a model loader: the most recently loaded model stays loaded.

    Wraps *loader* in ``functools.lru_cache(maxsize=1)`` — calling it
    again with the same arguments reuses the loaded model, calling it
    with different arguments (e.g. another model name or device) loads
    the new model and drops the previous one.  The loader is registered
    so :func:`clear_cache` clears it too.

    Loader arguments must be hashable and fully resolved (pass a real
    device string, not ``None``), since they form the cache key.
    """
    cached = lru_cache(maxsize=1)(loader)
    _MODEL_CACHES.append(cached)
    return cached


# ------------------------------------------------------------------
# MIS filter (cached loader + thin wrapper for cheap per-call
# settings like min_score and n_candidates)
# ------------------------------------------------------------------

@model_cache
def _load_mis_filter(device: str) -> MISFilter:
    """Load the MIS filter model (expensive, cached)."""
    return MISFilter(device=device)


def get_cached_mis_filter(
    device: str | None,
    **filter_kwargs: Any,
) -> MISFilter:
    """Return cached MIS filter, reloading only when *device* changes.

    Thin wrapper around :func:`_load_mis_filter`.  The model load is
    cached (expensive); this function just applies cheap per-call
    threshold settings on the existing instance.  Changing
    ``min_score`` or ``n_candidates`` between calls does not trigger a
    model reload — only a device change does.

    Parameters
    ----------
    device : str or None
        Torch device.  ``None`` resolves to :func:`default_device`.
    **filter_kwargs
        ``min_score`` and ``n_candidates``.  Missing keys reset to
        their defaults so that omitting a kwarg doesn't leave a stale
        value from a previous call.
    """
    device = device or default_device()
    mis_filter = _load_mis_filter(device)
    mis_filter.min_score = filter_kwargs.get("min_score", _DEFAULT_MIN_SCORE)
    mis_filter.n_candidates = filter_kwargs.get("n_candidates", _DEFAULT_N_CANDIDATES)
    return mis_filter


# ------------------------------------------------------------------
# Cache management
# ------------------------------------------------------------------

def clear_cache() -> None:
    """Drop references to all cached models so their memory can be reclaimed when possible.

    Clears every loader registered via :func:`model_cache` (the causal
    LM engine, the TinyStyler model, and the MIS filter).  After
    calling this, the next generation or filter use will load models
    from scratch.

    This clears Python-level references but does not guarantee immediate
    GPU/CPU memory release (e.g., allocator pools may retain reserved
    memory).
    """
    for cached_loader in _MODEL_CACHES:
        cached_loader.cache_clear()
