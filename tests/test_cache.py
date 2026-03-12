"""Tests for per-model caching in _cache.py."""

import unittest

import diversify_text._cache as _cache


class TestCache(unittest.TestCase):

    def setUp(self):
        _cache.clear_cache()

    def test_methods_cached_across_calls(self):
        methods_list_call_1 = _cache.get_methods(device=None, methods=["echo"])
        methods_list_call_2 = _cache.get_methods(device=None, methods=["echo"])
        self.assertIs(methods_list_call_1[0], methods_list_call_2[0])

    def test_mis_filter_created_with_default_device(self):
        """Regression: device=None must not collide with the unset sentinel."""
        mis_filter = _cache.get_mis_filter(device=None)
        self.assertIsNotNone(mis_filter)

    def test_mis_settings_update_without_reload(self):
        mis_filter = _cache.get_mis_filter(device=None, min_score=0.5)
        mis_filter_updated = _cache.get_mis_filter(device=None, min_score=0.9)
        self.assertIs(mis_filter, mis_filter_updated)
        self.assertEqual(mis_filter_updated.min_score, 0.9)

    def test_clear_cache_forces_new_instances(self):
        methods_list_before = _cache.get_methods(device=None, methods=["echo"])
        mis_filter_before = _cache.get_mis_filter(device=None)
        _cache.clear_cache()
        self.assertIsNot(_cache.get_methods(device=None, methods=["echo"])[0], methods_list_before[0])
        self.assertIsNot(_cache.get_mis_filter(device=None), mis_filter_before)


if __name__ == "__main__":
    unittest.main()
