"""Tests for the TinyStyler method's input checks."""

import unittest

from diversify_text.method.tinystyler import TinyStylerMethod


class TestTinyStylerInputChecks(unittest.TestCase):

    def test_n_larger_than_available_styles_raises(self):
        # No mock needed: the check runs before any model is loaded.
        # The default style list has 5 styles.
        method = TinyStylerMethod()
        with self.assertRaises(ValueError) as cm:
            method.generate(
                ["text"],
                n=7,
                max_new_tokens=None,
                temperature=None,
                top_p=None,
            )
        self.assertEqual(
            str(cm.exception),
            "n=7 exceeds the number of available styles (5).",
        )


if __name__ == "__main__":
    unittest.main()
