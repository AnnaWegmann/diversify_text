"""Per-model caching for the :func:`~diversify_text.core.diversify` convenience function.

Keeps the generation method(s) and the MIS filter in independent
module-level caches so that toggling ``semantic_filter`` does not
reload the generation model, and switching methods does not reload the
MIS model.

Each generation method is cached individually so that adding, removing,
or reordering methods only (re)loads the ones whose configuration
actually changed.

Not thread-safe.  Intended for single-threaded use in scripts and
notebooks.  For multi-threaded applications, use :class:`Diversifier`
directly with your own instance management.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

from diversify_text._utils import default_device
from diversify_text.filter.mis import MISFilter, _DEFAULT_MIN_SCORE, _DEFAULT_N_CANDIDATES
from diversify_text.method import DEFAULT_METHOD_REGISTRY, DiversificationMethod


# kwargs that affect model construction and should invalidate the cache.
# Per-call kwargs (styles, prompts, n_style_examples, etc.) are excluded.
_CACHE_KWARGS = {"model", "device", "precision"}


# ------------------------------------------------------------------
# Generation method cache (dict-based, one entry per method)
# ------------------------------------------------------------------

_METHOD_CACHE: dict[tuple, DiversificationMethod] = {}


def _resolve_cache_kwargs(
    method_name: str,
    device: str,
    method_kwargs: Mapping[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve the full set of cache-relevant kwargs for a method.

    Merges caller-provided kwargs with the constructor's own defaults
    (discovered via ``inspect.signature``) so that the cache key is
    the same whether the caller explicitly passes a default value or
    omits it.  Only kwargs in :data:`_CACHE_KWARGS` are included.

    For example, ``PromptingMethod.__init__`` has
    ``model="HuggingFaceTB/SmolLM3-3B"`` as a default.
    These two calls should hit the same cache entry::

        # Omit model — default is filled in from the signature.
        get_methods(device=None, methods=["prompting"])

        # Explicitly pass the same default value.
        get_methods(device=None, methods=["prompting"],
            method_kwargs={"prompting": {"model": "HuggingFaceTB/SmolLM3-3B"}})

    Without this function the first call would produce the key
    ``("prompting", (("device", "cpu"),))`` (no model) and the second
    ``("prompting", (("device", "cpu"), ("model", "HuggingFaceTB/..."),))``
    — different keys, two copies of the same model loaded.

    Parameters
    ----------
    method_name : str
        Registry name of the method (e.g. ``"tinystyler"``).
    device : str
        Torch device string (already resolved, never ``None``).
    method_kwargs : mapping, optional
        Per-method keyword arguments keyed by method name.  Only the
        entry for *method_name* is inspected.

    Returns
    -------
    dict[str, Any]
        The full set of cache-relevant kwargs, e.g.
        ``{"device": "cpu", "model": "HuggingFaceTB/SmolLM3-3B", "precision": "auto"}``.
    """
    # Start with device (always present).
    resolved: dict[str, Any] = {"device": device}

    # Fill in constructor defaults from the method class signature.
    method_class = DEFAULT_METHOD_REGISTRY.get(method_name)
    signature = inspect.signature(method_class)
    for param_name, param in signature.parameters.items():
        # inspect.Parameter.empty is a sentinel meaning "no default value."
        # We skip those — only fill in defaults that actually exist.
        if (
            param_name in _CACHE_KWARGS
            and param_name not in resolved
            and param.default is not inspect.Parameter.empty
        ):
            resolved[param_name] = param.default

    # Override defaults with caller-provided kwargs.
    if method_kwargs and (method_name in method_kwargs):
        for k, v in method_kwargs[method_name].items():
            if k in _CACHE_KWARGS:
                resolved[k] = v

    return resolved


def _single_METHOD_CACHE_key(
    method_name: str,
    device: str,
    method_kwargs: Mapping[str, dict[str, Any]] | None = None,
) -> tuple:
    """Build a hashable key for a single generation method.

    The key uniquely identifies a loaded model instance.  It includes
    only constructor-level kwargs (``model``, ``device``, ``precision``)
    that determine *which* model gets loaded.  Per-call kwargs like
    ``styles`` or ``prompts`` are excluded — changing those should reuse
    the same model, not trigger an expensive reload.

    Constructor defaults are resolved via ``inspect.signature`` so that
    explicitly passing a default value produces the same cache key as
    omitting it entirely.

    Parameters
    ----------
    method_name : str
        Registry name of the method (e.g. ``"tinystyler"``).
    device : str
        Torch device string (already resolved, never ``None``).
    method_kwargs : mapping, optional
        Per-method keyword arguments keyed by method name.  Only the
        entry for *method_name* is inspected, and only cache
        kwargs within that entry are included in the key.

    Returns
    -------
    tuple
        A hashable key, e.g.
        ``("prompting", "cpu", (("model", "default-model"), ("precision", "auto")))``.
    """
    resolved = _resolve_cache_kwargs(method_name, device, method_kwargs)
    constructor_kwargs = tuple(sorted(resolved.items()))
    return method_name, constructor_kwargs


def get_methods(
    device: str | None,
    methods: Sequence[str | DiversificationMethod] | None,
    method_kwargs: Mapping[str, dict[str, Any]] | None = None,
) -> list[DiversificationMethod]:
    """Return cached generation methods, resolving only on config change.

    Iterates the requested *methods* list and resolves each one
    individually against a module-level dict cache.  On a cache miss
    the method is instantiated via the registry (expensive — may load
    a model); on a hit the existing instance is reused.

    Methods can be specified as strings (looked up in the registry) or
    as pre-built :class:`DiversificationMethod` instances (passed
    through as-is without caching, since they're already instantiated).
    You can mix both in one call, e.g.
    ``methods=["tinystyler", my_custom_method]``.

    Because each method is cached independently, adding or removing a
    method from the list only loads the new ones — already-cached
    methods are not affected.

    Parameters
    ----------
    device : str or None
        Torch device.  ``None`` resolves to :func:`default_device`.
    methods : sequence of str or DiversificationMethod, optional
        Method names and/or pre-built instances.  Defaults to
        ``["tinystyler"]``.
    method_kwargs : mapping, optional
        Per-method keyword arguments keyed by method name, e.g.
        ``{"prompting": {"model": "gpt2"}}``.  Constructor kwargs
        (``model``, ``device``, ``precision``) affect the cache key;
        per-call kwargs (``styles``, ``prompts``) do not.

    Returns
    -------
    list[DiversificationMethod]
        Resolved method instances in the same order as *methods*.
    """
    device = device or default_device()
    if methods is None:
        methods = ["tinystyler"]

    result: list[DiversificationMethod] = []
    for method in methods:
        if isinstance(method, DiversificationMethod):
            result.append(method)
        elif isinstance(method, str):
            key = _single_METHOD_CACHE_key(method, device, method_kwargs)
            if key not in _METHOD_CACHE:  # cache miss → resolve and store
                resolve_kwargs: dict[str, Any] = {"device": device}
                if method_kwargs and (method in method_kwargs):
                    resolve_kwargs.update(method_kwargs[method])
                _METHOD_CACHE[key] = DEFAULT_METHOD_REGISTRY.resolve(
                    [method], **resolve_kwargs
                )[0]
            result.append(_METHOD_CACHE[key])
        else:
            raise TypeError(
                "method must be str or DiversificationMethod instance."
            )

    if not result:
        raise ValueError("At least one method is required.")
    return result


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

    Clears both the generation method dict cache and the ``lru_cache``
    backing the MIS filter.  After calling this, the next
    :func:`get_methods` or :func:`get_cached_mis_filter` call will
    load models from scratch.

    This clears Python-level references but does not guarantee immediate
    GPU/CPU memory release (e.g., allocator pools may retain reserved
    memory).
    """
    global _METHOD_CACHE

    _METHOD_CACHE = {}
    _load_mis_filter.cache_clear()
