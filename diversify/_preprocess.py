"""Text preprocessing utilities for diversify."""

from __future__ import annotations

import pysbd  # https://github.com/nipunsadvilkar/pySBD, published at EMLNP 2020, rule-based


_SEGMENTER = pysbd.Segmenter(language="en", clean=False)


def split_sentences(text: str) -> list[str]:
    """Split *text* into sentences using pysbd.

    Returns a list of stripped sentence strings.  If the text is empty or
    whitespace-only, returns a single-element list containing the stripped
    (possibly empty) input.
    """
    segments = _SEGMENTER.segment(text.strip())
    cleaned = [s.strip() for s in segments if s.strip()]
    return cleaned or [text.strip()]
