"""Tests for model caching: loaded models are cached, method instances are not."""

import unittest
import unittest.mock

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

    def test_clear_cache_forces_new_instance(self):
        mis_before = _cache.get_cached_mis_filter(device=None)
        _cache.clear_cache()
        mis_after = _cache.get_cached_mis_filter(device=None)
        self.assertIsNot(mis_before, mis_after)


# ------------------------------------------------------------------
# Causal LM engine cache (prompt-based methods)
# ------------------------------------------------------------------


class TestEngineCache(_CacheTestBase):

    @unittest.mock.patch("diversify_text.method.llm.PromptingModel.load")
    def test_same_configuration_shares_one_engine(self, _mock_load):
        """Two method instances with the same configuration share one loaded engine."""
        from diversify_text.method.prompting import PromptingMethod
        engine_a = PromptingMethod()._ensure_model()
        engine_b = PromptingMethod()._ensure_model()
        self.assertIs(engine_a, engine_b)

    @unittest.mock.patch("diversify_text.method.llm.PromptingModel.load")
    def test_clear_cache_drops_engines(self, _mock_load):
        from diversify_text.method.prompting import PromptingMethod
        engine_before = PromptingMethod()._ensure_model()
        _cache.clear_cache()
        engine_after = PromptingMethod()._ensure_model()
        self.assertIsNot(engine_before, engine_after)


# ------------------------------------------------------------------
# TinyStyler model cache
# ------------------------------------------------------------------


class TestTinyStylerModelCache(_CacheTestBase):

    @unittest.mock.patch(
        "diversify_text.method.tinystyler.model.TinyStyler._load_model",
        return_value=(None, None, None),
    )
    def test_same_device_shares_one_model(self, _mock_load):
        """Two method instances on the same device share one loaded model."""
        from diversify_text.method.tinystyler import TinyStylerMethod
        model_a = TinyStylerMethod()._ensure_model()
        model_b = TinyStylerMethod()._ensure_model()
        self.assertIs(model_a, model_b)

    @unittest.mock.patch(
        "diversify_text.method.tinystyler.model.TinyStyler._load_model",
        return_value=(None, None, None),
    )
    def test_clear_cache_drops_models(self, _mock_load):
        from diversify_text.method.tinystyler import TinyStylerMethod
        model_before = TinyStylerMethod()._ensure_model()
        _cache.clear_cache()
        model_after = TinyStylerMethod()._ensure_model()
        self.assertIsNot(model_before, model_after)


if __name__ == "__main__":
    unittest.main()
