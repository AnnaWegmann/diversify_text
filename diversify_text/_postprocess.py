"""Text postprocessing utilities for diversify."""

from __future__ import annotations

from diversify_text._preprocess import PreprocessContext


def reassemble_segments(
    segments_per_text: list[list[str]],
    paraphrases_by_segment: list[list[str]],
) -> list[list[str]]:
    """Join per-segment paraphrases back into per-original-text paraphrases.

    Parameters
    ----------
    segments_per_text : list[list[str]]
        The sentence segments for each original text (from
        :func:`~diversify_text._preprocess.split_sentences`).
    paraphrases_by_segment : list[list[str]]
        Flat list of paraphrases for every segment, shape
        ``[total_segments][n]``.

    Returns
    -------
    list[list[str]]
        Shape ``[n_texts][n]`` — reassembled paraphrases.
    """
    result = []
    seg_idx = 0
    for segs in segments_per_text:
        seg_paras = paraphrases_by_segment[seg_idx : seg_idx + len(segs)]
        n = len(seg_paras[0])
        result.append([" ".join(sp[i] for sp in seg_paras) for i in range(n)])
        seg_idx += len(segs)
    return result


def postprocess(
    candidate: list[list[str]],
    context: PreprocessContext,
) -> list[list[str]]:
    """Undo preprocessing transformations on a candidate set.

    Applies the inverse of each step performed by
    :func:`~diversify_text._preprocess.preprocess`, using the state stored in
    *context*.

    Parameters
    ----------
    candidate : list[list[str]]
        Raw generation output, shape ``[n_generation_texts][n]``.
    context : PreprocessContext
        Context returned by :func:`~diversify_text._preprocess.preprocess`.

    Returns
    -------
    list[list[str]]
        Shape ``[n_texts][n]`` — one paraphrase per original text
        per style.
    """
    if context.segments_per_text is not None:
        candidate = reassemble_segments(context.segments_per_text, candidate)

    return candidate
