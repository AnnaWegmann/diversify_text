"""Text preprocessing utilities for diversify."""

from __future__ import annotations

from dataclasses import dataclass

import pysbd  # https://github.com/nipunsadvilkar/pySBD, published at EMNLP 2020, rule-based


_SEGMENTER = pysbd.Segmenter(language="en", clean=False)


@dataclass
class PreprocessContext:
    """State produced by :func:`preprocess` and consumed by
    :func:`~diversify_text._postprocess.postprocess`.

    New preprocessing steps can add fields here without changing the
    caller in ``core.py``.
    """

    segments_per_text: list[list[str]] | None = None


def split_sentences(text: str) -> list[str]:
    """Split *text* into sentences using pysbd.

    Returns a list of stripped sentence strings.  If the text is empty or
    whitespace-only, returns a single-element list containing the stripped
    (possibly empty) input.
    """
    segments = _SEGMENTER.segment(text.strip())
    cleaned = [s.strip() for s in segments if s.strip()]
    return cleaned or [text.strip()]


def preprocess(
    texts: list[str],
    *,
    split_on_punctuation: bool = False,
) -> tuple[list[str], PreprocessContext]:
    """Prepare a batch of texts for generation.

    Returns the (possibly transformed) texts to feed into the generation
    method, together with a :class:`PreprocessContext` that
    :func:`~diversify_text._postprocess.postprocess` needs to undo the
    transformations.

    Parameters
    ----------
    texts : list[str]
        Original input texts.
    split_on_punctuation : bool
        If ``True``, split each text into sentence-level segments and
        flatten the result.  The per-text segment mapping is stored in
        the context so that :func:`~diversify_text._postprocess.postprocess`
        can reassemble them.

    Returns
    -------
    generation_texts : list[str]
        Texts to pass to the generation method.
    context : PreprocessContext
        Context needed by :func:`~diversify_text._postprocess.postprocess`.
    """
    context = PreprocessContext()

    if split_on_punctuation:
        context.segments_per_text = [split_sentences(t) for t in texts]
        generation_texts = [
            seg for segs in context.segments_per_text for seg in segs
        ]
    else:
        generation_texts = texts

    return generation_texts, context
