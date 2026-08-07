"""Model caching.

Loaded models are cached, method instances are not.  The rule: method
objects must be cheap to create, and anything expensive — model
loading — happens inside a *loader*, a small function whose only job
is to load and return one model, wrapped with :func:`model_cache`.
The loaders live next to their model classes: the causal language
model in :mod:`diversify_text.method.llm`, the TinyStyler model in
:mod:`diversify_text.method.tinystyler.model`, and the MIS filter
below.

Not thread-safe.  Intended for single-threaded use in scripts and
notebooks.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

from diversify_text._utils import default_device
from diversify_text.filter.mis import MISFilter, _DEFAULT_MIN_SCORE, _DEFAULT_N_CANDIDATES

#: All cached loaders, registered by :func:`model_cache`.
_MODEL_CACHES: list[Any] = []


def model_cache(loader: Callable) -> Callable:
    """Cache a model loader: the most recently loaded model stays loaded.

    A *loader* is a small function whose only job is to load and return
    one model.  This decorator wraps it in
    ``functools.lru_cache(maxsize=1)`` and registers it so
    :func:`clear_cache` clears it too.  Each decorated loader gets its
    own independent cache — one causal LM, one TinyStyler and one MIS
    filter can all be loaded at the same time; only loading a *second*
    configuration of the same family drops the previous one.

    The loader's parameters are the cache key — nothing more.  A loader
    therefore takes exactly the arguments that determine which model
    gets loaded (hashable and fully resolved: pass a real device
    string, not ``None``).  Settings that should not cause a reload are
    applied after loading, outside the loader (see
    :func:`get_cached_mis_filter`).  Loaders are standalone functions
    rather than methods on the classes, because a method's ``self``
    would be part of the key: every instance would get its own cache
    entry, and no sharing would happen.
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
