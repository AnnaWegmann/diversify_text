"""Greedy maxmin diversity selection for style embeddings."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_N_CANDIDATES = 100


class DiverseEmbeddingSelector:
    """Select style embeddings that maximize minimum pairwise cosine distance.

    For each call to :meth:`select`, generates a pool of random vectors
    sampled from the same distribution as real style embeddings and
    greedily picks the most diverse ones relative to all previously
    selected embeddings.  Accumulates state across calls so that later
    texts receive embeddings different from earlier ones.

    Parameters
    ----------
    mean : torch.Tensor
        Per-dimension mean of real style embeddings, shape ``(dim,)``.
    std : torch.Tensor
        Per-dimension std of real style embeddings, shape ``(dim,)``.
    max_pool_size : int
        Maximum number of embeddings to accumulate.  Once reached, new
        calls to :meth:`select` reuse existing embeddings instead of
        generating new ones.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        max_pool_size: int = 1000,
    ) -> None:
        self._mean = mean
        self._std = std
        self._max_pool_size = max_pool_size
        self._selected: list[torch.Tensor] = []

    def reset(self) -> None:
        """Clear accumulated embeddings."""
        self._selected.clear()

    def select(self, n: int, device: torch.device | str) -> list[torch.Tensor]:
        """Select *n* diverse style embeddings.

        If the pool has not yet reached *max_pool_size*, generates 100
        random candidate vectors (matching the style embedding distribution)
        and greedily picks the most diverse ones.  Once the cap is reached,
        cycles through existing embeddings.

        Returns
        -------
        list[torch.Tensor]
            *n* tensors of shape ``(1, embedding_dim)``.
        """
        if len(self._selected) >= self._max_pool_size:
            return self._reuse(n)

        remaining = self._max_pool_size - len(self._selected)
        n_new = min(n, remaining)
        n_reuse = n - n_new

        candidates = _random_style_vectors(
            _N_CANDIDATES, self._mean, self._std, device,
        )
        indices = _greedy_maxmin(candidates, self._selected, n_new)
        new_embeddings = [candidates[i].unsqueeze(0) for i in indices]
        self._selected.extend(new_embeddings)

        if n_reuse > 0:
            reused = self._reuse(n_reuse)
            return new_embeddings + reused

        return new_embeddings

    def _reuse(self, n: int) -> list[torch.Tensor]:
        """Cycle through existing pool embeddings."""
        pool_size = len(self._selected)
        return [self._selected[i % pool_size] for i in range(n)]


def compute_style_stats(
    model: object,
    style_bank: dict[str, list[str]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-dimension mean and std from real style embeddings.

    Parameters
    ----------
    model : TinyStyler
        Model instance with a ``get_style_embedding`` method.
    style_bank : dict[str, list[str]]
        Mapping of style name to example texts.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(mean, std)`` each of shape ``(dim,)``.
    """
    embeddings = []
    for examples in style_bank.values():
        emb = model.get_style_embedding(examples)  # (1, dim)
        embeddings.append(emb.squeeze(0))
    stacked = torch.stack(embeddings)  # (n_styles, dim)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0)
    # Avoid zero std (constant dimensions) — clamp to a small floor.
    std = std.clamp(min=1e-6)
    logger.info(
        "Computed style embedding stats from %d styles (dim=%d).",
        len(embeddings),
        mean.shape[0],
    )
    return mean, std


def _random_style_vectors(
    n: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device | str,
) -> torch.Tensor:
    """Generate *n* random vectors matching the style embedding distribution.

    Samples from ``N(mean, std)`` per dimension.

    Returns a tensor of shape ``(n, dim)``.
    """
    dim = mean.shape[0]
    vecs = torch.randn(n, dim, device=device)
    return vecs * std.to(device) + mean.to(device)


def _greedy_maxmin(
    candidates: torch.Tensor,
    existing: list[torch.Tensor],
    n_select: int,
) -> list[int]:
    """Greedy farthest-point selection using cosine distance.

    Parameters
    ----------
    candidates : torch.Tensor
        Shape ``(n_candidates, dim)``.
    existing : list[torch.Tensor]
        Previously selected embeddings, each of shape ``(1, dim)``.
    n_select : int
        Number of candidates to select.

    Returns
    -------
    list[int]
        Indices into *candidates* of the selected points.
    """
    n_cand = candidates.shape[0]
    if n_select >= n_cand:
        return list(range(n_cand))

    # Normalise for cosine distance computation.
    cand_norm = F.normalize(candidates, dim=-1)

    # Build matrix of existing embeddings for distance computation.
    if existing:
        existing_mat = torch.cat(existing, dim=0)  # (n_existing, dim)
        existing_norm = F.normalize(existing_mat, dim=-1)
    else:
        existing_norm = None

    selected_indices: list[int] = []
    # Track min distance from each candidate to the closest selected point.
    if existing_norm is not None:
        cos_sim = cand_norm @ existing_norm.T  # (n_cand, n_existing)
        min_dist = (1.0 - cos_sim).min(dim=1).values  # (n_cand,)
    else:
        min_dist = torch.full((n_cand,), float("inf"), device=candidates.device)

    for _ in range(n_select):
        # Pick candidate with largest min-distance.
        best_idx = min_dist.argmax().item()
        selected_indices.append(best_idx)

        # Update min_dist: compare each candidate to the newly selected point.
        new_sim = cand_norm @ cand_norm[best_idx]  # (n_cand,)
        new_dist = 1.0 - new_sim
        min_dist = torch.minimum(min_dist, new_dist)

        # Ensure already-selected candidates are not picked again.
        min_dist[best_idx] = -float("inf")

    return selected_indices
