"""Tests for per-model caching in _cache.py."""

import unittest

from diversify_text._cache import clear_cache, get_methods, get_mis_filter


class TestCache(unittest.TestCase):

    def setUp(self):
        clear_cache()

    def test_methods_cached_across_calls(self):
        methods_list_call_1 = get_methods(None, ["echo"])
        methods_list_call_2 = get_methods(None, ["echo"])
        self.assertIs(methods_list_call_1[0], methods_list_call_2[0])

    def test_mis_filter_created_with_default_device(self):
        """Regression: device=None must not collide with the unset sentinel."""
        mis_filter = get_mis_filter(None)
        self.assertIsNotNone(mis_filter)

    def test_mis_settings_update_without_reload(self):
        mis_filter = get_mis_filter(None, min_score=0.5)
        mis_filter_updated = get_mis_filter(None, min_score=0.9)
        self.assertIs(mis_filter, mis_filter_updated)
        self.assertEqual(mis_filter_updated.min_score, 0.9)

    def test_clear_cache_forces_new_instances(self):
        methods_list_before = get_methods(None, ["echo"])
        mis_filter_before = get_mis_filter(None)
        clear_cache()
        self.assertIsNot(get_methods(None, ["echo"])[0], methods_list_before[0])
        self.assertIsNot(get_mis_filter(None), mis_filter_before)


if __name__ == "__main__":
    unittest.main()
