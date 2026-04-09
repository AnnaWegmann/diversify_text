"""Tests for per-model caching in _cache.py."""

import unittest

import diversify_text._cache as _cache


class _CacheTestBase(unittest.TestCase):
    """Shared setUp that clears the cache before each test."""

    def setUp(self):
        _cache.clear_cache()


# ------------------------------------------------------------------
# MIS filter cache
# ------------------------------------------------------------------


class TestMISFilterCache(_CacheTestBase):

    def test_created_with_default_device(self):
        """Regression: device=None must not collide with the unset sentinel."""
        mis_filter = _cache.get_cached_mis_filter(device=None)
        self.assertIsNotNone(mis_filter)

    def test_cached_across_calls(self):
        mis1 = _cache.get_cached_mis_filter(device=None)
        mis2 = _cache.get_cached_mis_filter(device=None)
        self.assertIs(mis1, mis2)

    def test_settings_update_without_reload(self):
        mis_filter = _cache.get_cached_mis_filter(device=None, min_score=0.5)
        mis_filter_updated = _cache.get_cached_mis_filter(device=None, min_score=0.9)
        self.assertIs(mis_filter, mis_filter_updated)
        self.assertEqual(mis_filter_updated.min_score, 0.9)

    def test_reloads_on_device_change(self):
        mis_cpu = _cache.get_cached_mis_filter(device="cpu")
        mis_cpu2 = _cache.get_cached_mis_filter(device="cpu")
        self.assertIs(mis_cpu, mis_cpu2)

        mis_other = _cache.get_cached_mis_filter(device="meta")
        self.assertIsNot(mis_cpu, mis_other)

    def test_not_affected_by_method_changes(self):
        """Switching methods should not reload the MIS filter."""
        mis_filter = _cache.get_cached_mis_filter(device=None)
        _cache.get_methods(device=None, methods=["tinystyler"])
        _cache.get_methods(device=None, methods=["prompting"])
        mis_filter_after = _cache.get_cached_mis_filter(device=None)
        self.assertIs(mis_filter, mis_filter_after)

    def test_clear_cache_forces_new_instance(self):
        mis_before = _cache.get_cached_mis_filter(device=None)
        _cache.clear_cache()
        mis_after = _cache.get_cached_mis_filter(device=None)
        self.assertIsNot(mis_before, mis_after)


# ------------------------------------------------------------------
# TinyStyler method cache
# ------------------------------------------------------------------


class TestTinyStylerCache(_CacheTestBase):

    def test_cached_across_calls(self):
        [ts1] = _cache.get_methods(device=None, methods=["tinystyler"])
        [ts2] = _cache.get_methods(device=None, methods=["tinystyler"])
        self.assertIs(ts1, ts2)

    def test_changing_styles_does_not_reload(self):
        """styles is a per-call kwarg, not a constructor kwarg."""
        [ts1] = _cache.get_methods(
            device=None, methods=["tinystyler"],
            method_kwargs={"tinystyler": {"styles": ["formal"]}},
        )
        [ts2] = _cache.get_methods(
            device=None, methods=["tinystyler"],
            method_kwargs={"tinystyler": {"styles": ["casual"]}},
        )
        self.assertIs(ts1, ts2)

    def test_not_affected_by_mis_filter_changes(self):
        [ts] = _cache.get_methods(device=None, methods=["tinystyler"])
        _cache.get_cached_mis_filter(device=None, min_score=0.5)
        _cache.get_cached_mis_filter(device=None, min_score=0.9)
        [ts_after] = _cache.get_methods(device=None, methods=["tinystyler"])
        self.assertIs(ts, ts_after)

    def test_clear_cache_forces_new_instance(self):
        [ts_before] = _cache.get_methods(device=None, methods=["tinystyler"])
        _cache.clear_cache()
        [ts_after] = _cache.get_methods(device=None, methods=["tinystyler"])
        self.assertIsNot(ts_before, ts_after)


# ------------------------------------------------------------------
# Prompting method cache
# ------------------------------------------------------------------


class TestPromptingCache(_CacheTestBase):

    def test_cached_across_calls(self):
        [p1] = _cache.get_methods(device=None, methods=["prompting"])
        [p2] = _cache.get_methods(device=None, methods=["prompting"])
        self.assertIs(p1, p2)

    def test_changing_prompts_does_not_reload(self):
        """prompts is a per-call kwarg, not a constructor kwarg."""
        [p1] = _cache.get_methods(
            device=None, methods=["prompting"],
            method_kwargs={"prompting": {"prompts": ["Rewrite formally."]}},
        )
        [p2] = _cache.get_methods(
            device=None, methods=["prompting"],
            method_kwargs={"prompting": {"prompts": ["Rewrite casually."]}},
        )
        self.assertIs(p1, p2)

    def test_changing_model_reloads(self):
        """model is a constructor kwarg — changing it must reload."""
        [p1] = _cache.get_methods(
            device=None, methods=["prompting"],
            method_kwargs={"prompting": {"model": "model-a"}},
        )
        [p2] = _cache.get_methods(
            device=None, methods=["prompting"],
            method_kwargs={"prompting": {"model": "model-b"}},
        )
        self.assertIsNot(p1, p2)

    def test_changing_precision_reloads(self):
        """precision is a constructor kwarg — changing it must reload."""
        [p1] = _cache.get_methods(
            device=None, methods=["prompting"],
            method_kwargs={"prompting": {"precision": "float16"}},
        )
        [p2] = _cache.get_methods(
            device=None, methods=["prompting"],
            method_kwargs={"prompting": {"precision": "bfloat16"}},
        )
        self.assertIsNot(p1, p2)

    def test_explicit_default_reuses_cache(self):
        """Passing the default model explicitly should hit the same cache entry."""
        from diversify_text.method.prompting.method import _DEFAULT_MODEL
        [p1] = _cache.get_methods(device=None, methods=["prompting"])
        [p2] = _cache.get_methods(
            device=None, methods=["prompting"],
            method_kwargs={"prompting": {"model": _DEFAULT_MODEL}},
        )
        self.assertIs(p1, p2)

    def test_not_affected_by_mis_filter_changes(self):
        [p] = _cache.get_methods(device=None, methods=["prompting"])
        _cache.get_cached_mis_filter(device=None, min_score=0.5)
        _cache.get_cached_mis_filter(device=None, min_score=0.9)
        [p_after] = _cache.get_methods(device=None, methods=["prompting"])
        self.assertIs(p, p_after)

    def test_clear_cache_forces_new_instance(self):
        [p_before] = _cache.get_methods(device=None, methods=["prompting"])
        _cache.clear_cache()
        [p_after] = _cache.get_methods(device=None, methods=["prompting"])
        self.assertIsNot(p_before, p_after)


# ------------------------------------------------------------------
# Multiple methods cached together
# ------------------------------------------------------------------


class TestMultiMethodCache(_CacheTestBase):

    def test_adding_method_reuses_existing(self):
        """Adding a method to the list should not reload already-cached methods."""
        [ts1] = _cache.get_methods(device=None, methods=["tinystyler"])
        ts2, _p = _cache.get_methods(device=None, methods=["tinystyler", "prompting"])
        self.assertIs(ts1, ts2)

    def test_removing_method_preserves_remaining(self):
        """Removing a method from the list should not reload the remaining ones."""
        ts1, _p = _cache.get_methods(device=None, methods=["tinystyler", "prompting"])
        [ts2] = _cache.get_methods(device=None, methods=["tinystyler"])
        self.assertIs(ts1, ts2)

    def test_order_preserved(self):
        """Different orderings should reuse the same instances."""
        ts1, p1 = _cache.get_methods(device=None, methods=["tinystyler", "prompting"])
        p2, ts2 = _cache.get_methods(device=None, methods=["prompting", "tinystyler"])
        self.assertIs(ts1, ts2)
        self.assertIs(p1, p2)


# ------------------------------------------------------------------
# Pre-built instances
# ------------------------------------------------------------------


class TestPrebuiltInstanceCache(_CacheTestBase):

    def test_passthrough_without_caching(self):
        """Pre-built instances should be returned as-is and not stored in cache."""
        from diversify_text.method.echo import EchoMethod
        instance = EchoMethod()
        result = _cache.get_methods(device=None, methods=[instance])
        self.assertIs(result[0], instance)
        for cached in _cache._METHOD_CACHE.values():
            self.assertIsNot(cached, instance)


if __name__ == "__main__":
    unittest.main()
