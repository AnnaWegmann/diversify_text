"""diversify -- generate stylistic paraphrases of texts."""

import logging

from diversify.core import (
    Diversifier,
    diversify,
)

__all__ = [
    "Diversifier",
    "diversify",
]

# Configure a clean handler for the diversify logger so INFO/WARNING messages
# are visible without requiring the user to set up logging themselves.
_logger = logging.getLogger("diversify")
_logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
_logger.addHandler(_handler)
# Prevent messages from bubbling up to the root logger (avoids duplicate output
# if the user has already configured logging globally).
_logger.propagate = False
