"""diversify-text -- generate stylistic paraphrases of texts."""

import logging

from diversify_text._cache import clear_cache
from diversify_text.core import (
    Diversifier,
    diversify,
)

__all__ = [
    "Diversifier",
    "clear_cache",
    "diversify",
]

# Configure a clean handler for the diversify logger so INFO/WARNING messages
# are visible without requiring the user to set up logging themselves.
_logger = logging.getLogger("diversify_text")
_logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
_logger.addHandler(_handler)
# Prevent messages from bubbling up to the root logger (avoids duplicate output
# if the user has already configured logging globally).
_logger.propagate = False
