"""Shared internal utilities for diversify."""

from __future__ import annotations

import contextlib
import logging
import warnings


def default_device() -> str:
    """Return the best available torch device (``"cuda"``, ``"mps"``, or ``"cpu"``)."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@contextlib.contextmanager
def suppress_hf_load_noise():
    """Silence harmless noise emitted when loading HuggingFace models.

    Covers two sources that Python's warnings module alone cannot reach:

    - Tied-weights notices from the ``transformers`` logging system.
    - Unexpected-key load reports from the style-embedding model.
    """
    transformers_logger = logging.getLogger("transformers")
    prev_level = transformers_logger.level
    transformers_logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*tie.*weight.*")
            yield
    finally:
        transformers_logger.setLevel(prev_level)
