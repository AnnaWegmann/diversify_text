"""Model caching for the :func:`~diversify_text.core.diversify` convenience function.

Loaded models are cached, method instances are not: method objects are
cheap to create, and each model family keeps one loaded instance per
configuration in its own module (the causal LM engine in
:mod:`diversify_text.method.llm`, the TinyStyler model in
:mod:`diversify_text.method.tinystyler.model`, the MIS filter here).
Toggling ``semantic_filter`` therefore does not reload the generation
model, switching methods does not reload the MIS model, and per-call
options never trigger a model reload.

Not thread-safe.  Intended for single-threaded use in scripts and
notebooks.  For multi-threaded applications, use :class:`Diversifier`
directly with your own instance management.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from diversify_text._utils import default_device
from diversify_text.filter.mis import MISFilter, _DEFAULT_MIN_SCORE, _DEFAULT_N_CANDIDATES
from diversify_text.method import DEFAULT_METHOD_REGISTRY, DiversificationMethod
from diversify_text.method.llm import clear_engine_cache
from diversify_text.method.tinystyler.model import clear_model_cache


def get_method(
    device: str | None,
    method: str | DiversificationMethod | None,
    method_kwargs: Mapping[str, Any] | None = None,
) -> DiversificationMethod:
    """Resolve *method* into a ready instance.

    Method instances are cheap and are created fresh on every call —
    the expensive model loading is cached at the model level in each
    method's module, so a fresh instance still reuses loaded models.

    Parameters
    ----------
    device : str or None
        Torch device.  ``None`` lets the method auto-detect.
    method : str or DiversificationMethod, optional
        Method name or pre-built instance (returned as-is).  Defaults
        to ``"tinystyler"``.
    method_kwargs : mapping, optional
        Method-specific keyword arguments, e.g. ``{"model": "gpt2"}``.
        Constructor arguments are forwarded to the method; per-call
        arguments (``prompt``, ...) are ignored here and applied at
        generation time.
    """
    if isinstance(method, DiversificationMethod):
        return method
    if method is None:
        method = "tinystyler"
    if isinstance(method, str):
        resolve_kwargs: dict[str, Any] = {"device": device}
        if method_kwargs:
            resolve_kwargs.update(method_kwargs)
        return DEFAULT_METHOD_REGISTRY.resolve(method, **resolve_kwargs)
    raise TypeError(
        "method must be str or DiversificationMethod instance."
    )


# ------------------------------------------------------------------
# MIS filter cache (lru_cache for expensive model load, thin wrapper
# for cheap per-call settings like min_score and n_candidates)
# ------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_mis_filter(device: str) -> MISFilter:
    """Load the MIS filter model (expensive).

    This is the expensive part — loading the model weights.  The
    ``lru_cache`` decorator ensures this only runs once per last used device
    string.  Cheap per-call settings (``min_score``, ``n_candidates``)
    are applied separately in :func:`get_cached_mis_filter`.
    """
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

    Clears the causal LM engine cache, the TinyStyler model cache, and
    the ``lru_cache`` backing the MIS filter.  After calling this, the
    next generation or filter use will load models from scratch.

    This clears Python-level references but does not guarantee immediate
    GPU/CPU memory release (e.g., allocator pools may retain reserved
    memory).
    """
    clear_engine_cache()
    clear_model_cache()
    _load_mis_filter.cache_clear()
