"""Tests for DiverseEmbeddingSelector and maximize_CISR_diversity integration."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from diversify_text.method.tinystyler.diversity import (
    DiverseEmbeddingSelector,
    _greedy_maxmin,
    _random_style_vectors,
)

# Helpers for creating test mean/std tensors.
_DIM = 64
_MEAN = torch.randn(_DIM)
_STD = torch.rand(_DIM).clamp(min=0.1)


def _make_selector(**kwargs) -> DiverseEmbeddingSelector:
    return DiverseEmbeddingSelector(mean=_MEAN, std=_STD, **kwargs)


# ------------------------------------------------------------------
# Unit tests: random style vectors
# ------------------------------------------------------------------


class TestRandomStyleVectors:
    def test_shape(self):
        vecs = _random_style_vectors(10, _MEAN, _STD, "cpu")
        assert vecs.shape == (10, _DIM)

    def test_distribution_matches(self):
        """Generated vectors should have similar mean/std to the input."""
        torch.manual_seed(0)
        vecs = _random_style_vectors(10_000, _MEAN, _STD, "cpu")
        assert torch.allclose(vecs.mean(dim=0), _MEAN, atol=0.1)
        assert torch.allclose(vecs.std(dim=0), _STD, atol=0.1)


# ------------------------------------------------------------------
# Unit tests: greedy maxmin
# ------------------------------------------------------------------


class TestGreedyMaxmin:
    def test_selects_correct_count(self):
        candidates = _random_style_vectors(100, _MEAN, _STD, "cpu")
        indices = _greedy_maxmin(candidates, [], 5)
        assert len(indices) == 5
        assert len(set(indices)) == 5  # all unique

    def test_with_existing_embeddings(self):
        torch.manual_seed(42)
        candidates = _random_style_vectors(100, _MEAN, _STD, "cpu")
        existing = [_random_style_vectors(1, _MEAN, _STD, "cpu") for _ in range(3)]
        indices = _greedy_maxmin(candidates, existing, 5)
        assert len(indices) == 5

    def test_returns_all_when_n_select_ge_candidates(self):
        candidates = _random_style_vectors(5, _MEAN, _STD, "cpu")
        indices = _greedy_maxmin(candidates, [], 10)
        assert len(indices) == 5


# ------------------------------------------------------------------
# Unit tests: DiverseEmbeddingSelector
# ------------------------------------------------------------------


class TestDiverseEmbeddingSelector:
    def test_select_returns_correct_count_and_shape(self):
        selector = _make_selector()
        embeddings = selector.select(5, "cpu")
        assert len(embeddings) == 5
        for emb in embeddings:
            assert emb.shape == (1, _DIM)

    def test_accumulation_across_calls(self):
        selector = _make_selector()
        selector.select(3, "cpu")
        assert len(selector._selected) == 3
        selector.select(4, "cpu")
        assert len(selector._selected) == 7

    def test_reset_clears_state(self):
        selector = _make_selector()
        selector.select(5, "cpu")
        assert len(selector._selected) == 5
        selector.reset()
        assert len(selector._selected) == 0

    def test_cap_at_max_pool_size(self):
        selector = _make_selector(max_pool_size=10)
        selector.select(8, "cpu")
        assert len(selector._selected) == 8
        # Request 5 more, but only 2 slots remain → 2 new + 3 reused
        result = selector.select(5, "cpu")
        assert len(result) == 5
        assert len(selector._selected) == 10  # capped

    def test_reuse_when_full(self):
        selector = _make_selector(max_pool_size=5)
        selector.select(5, "cpu")
        assert len(selector._selected) == 5
        # Pool is full, should reuse
        result = selector.select(3, "cpu")
        assert len(result) == 3
        assert len(selector._selected) == 5  # no growth

    def test_deterministic_with_seed(self):
        torch.manual_seed(42)
        sel1 = _make_selector()
        embs1 = sel1.select(5, "cpu")

        torch.manual_seed(42)
        sel2 = _make_selector()
        embs2 = sel2.select(5, "cpu")

        for a, b in zip(embs1, embs2):
            assert torch.allclose(a, b)


# ------------------------------------------------------------------
# Integration tests: TinyStylerMethod with maximize_CISR_diversity
# ------------------------------------------------------------------


class TestMaximizeCISRDiversityIntegration:
    def _make_mock_model(self, embedding_dim=64):
        """Create a mock TinyStyler that returns dummy outputs."""
        model = MagicMock()
        model.device = "cpu"
        # get_style_embedding returns a tensor of the right shape
        model.get_style_embedding.return_value = torch.randn(1, embedding_dim)
        # transfer returns dummy text for each input
        model.transfer.side_effect = lambda texts, style, **kw: [
            f"paraphrased:{t}" for t in texts
        ]
        # tokenizer for max_new_tokens computation
        model._tokenizer.return_value = {"input_ids": [[1, 2, 3]]}
        return model

    def test_generate_with_maximize_CISR_diversity(self):
        from diversify_text.method.tinystyler.method import TinyStylerMethod

        method = TinyStylerMethod(device="cpu")
        mock_model = self._make_mock_model()
        method._model = mock_model

        result = method.generate(
            ["hello", "world"],
            n_styles=3,
            max_new_tokens=None,
            temperature=None,
            top_p=None,
            maximize_CISR_diversity=True,
        )

        # 2 texts, 3 styles each
        assert len(result) == 2
        assert len(result[0]) == 3
        assert len(result[1]) == 3

        # transfer() should have been called with tensor styles
        for call in mock_model.transfer.call_args_list:
            style_arg = call[0][1]
            assert isinstance(style_arg, torch.Tensor)

    def test_reset_clears_diversity_state(self):
        from diversify_text.method.tinystyler.method import TinyStylerMethod

        method = TinyStylerMethod(device="cpu")
        mock_model = self._make_mock_model()
        method._model = mock_model

        method.generate(
            ["text"],
            n_styles=3,
            max_new_tokens=None,
            temperature=None,
            top_p=None,
            maximize_CISR_diversity=True,
        )
        assert method._diversity_selector is not None
        assert len(method._diversity_selector._selected) == 3

        method.reset()
        assert len(method._diversity_selector._selected) == 0

    def test_accumulation_across_batches(self):
        from diversify_text.method.tinystyler.method import TinyStylerMethod

        method = TinyStylerMethod(device="cpu")
        mock_model = self._make_mock_model()
        method._model = mock_model

        method.generate(
            ["text1"],
            n_styles=3,
            max_new_tokens=None,
            temperature=None,
            top_p=None,
            maximize_CISR_diversity=True,
        )
        assert len(method._diversity_selector._selected) == 3

        method.generate(
            ["text2"],
            n_styles=3,
            max_new_tokens=None,
            temperature=None,
            top_p=None,
            maximize_CISR_diversity=True,
        )
        assert len(method._diversity_selector._selected) == 6
