"""
Core module for text diversification via stylistic paraphrasing.

Provides the :class:`Diversifier` class and a convenience :func:`diversify`
function that produce multiple stylistically varied paraphrases for each
input text.
"""

from __future__ import annotations

from typing import Union

import pandas as pd


TextInput = Union[str, list[str], pd.Series, pd.DataFrame]


class Diversifier:
    """Generate stylistic paraphrases of texts using local transformer models.

    The model is loaded once on instantiation and reused across calls.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace model identifier or local path.  Defaults to the
        TinyStyler model.
    device : str, optional
        Torch device (``"cuda"``, ``"cpu"``, ``"mps"``, ...).
        Auto-detected when *None*.

    Example
    -------
    >>> div = Diversifier()
    >>> results = div.diversify("The experiment was conducted in a lab.")
    >>> len(results)  # one dict per input text
    1
    >>> list(results[0].keys())
    ['original', 'paraphrases']
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None  # lazy-loaded on first call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diversify(
        self,
        texts: TextInput,
        *,
        n_styles: int = 5,
        text_column: str = "text",
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> list[dict]:
        """Produce *n_styles* stylistic paraphrases for each input text.

        Parameters
        ----------
        texts : str | list[str] | pd.Series | pd.DataFrame
            Input text(s).  When a :class:`~pandas.DataFrame` is passed,
            the column specified by *text_column* is used.
        n_styles : int
            Number of stylistically diverse paraphrases to generate per
            input text.
        text_column : str
            Column name to read when *texts* is a DataFrame.
        max_new_tokens : int
            Maximum number of tokens to generate per paraphrase.
        temperature : float
            Sampling temperature.
        top_p : float
            Nucleus-sampling probability mass.

        Returns
        -------
        list[dict]
            A list with one entry per input text.  Each entry is a dict::

                {
                    "original": str,
                    "paraphrases": list[str],   # length == n_styles
                }
        """
        text_list = self._normalize_input(texts, text_column)

        self._ensure_model_loaded()

        # TODO: implement actual model-based diversification.
        #  The stub below returns the originals as placeholders.
        results = []
        for text in text_list:
            results.append(
                {
                    "original": text,
                    "paraphrases": [text] * n_styles,  # placeholder
                }
            )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_input(self, texts: TextInput, text_column: str) -> list[str]:
        """Coerce any supported input type into ``list[str]``."""
        if isinstance(texts, str):
            return [texts]
        if isinstance(texts, pd.Series):
            return texts.tolist()
        if isinstance(texts, pd.DataFrame):
            return texts[text_column].tolist()
        if isinstance(texts, list):
            return texts
        raise TypeError(
            f"Unsupported input type {type(texts).__name__}. "
            "Expected str, list[str], pd.Series, or pd.DataFrame."
        )

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the model on first use."""
        if self._model is not None:
            return
        # TODO: load the actual model here.
        #  For now this is a no-op so that the API surface can be used
        #  and tested without downloading model weights.
        self._model = "placeholder"


# ------------------------------------------------------------------
# Module-level convenience function
# ------------------------------------------------------------------


def diversify(
    texts: TextInput,
    *,
    model_name: str | None = None,
    device: str | None = None,
    **kwargs,
) -> list[dict]:
    """One-shot convenience function: create a :class:`Diversifier` and run it.

    Parameters
    ----------
    texts : str | list[str] | pd.Series | pd.DataFrame
        Input text(s).
    model_name : str, optional
        HuggingFace model identifier or local path.
    device : str, optional
        Torch device.
    **kwargs
        Forwarded to :meth:`Diversifier.diversify`
        (``n_styles``, ``text_column``, ``max_new_tokens``,
        ``temperature``, ``top_p``).

    Returns
    -------
    list[dict]
        See :meth:`Diversifier.diversify`.
    """
    div = Diversifier(model_name=model_name, device=device)
    return div.diversify(texts, **kwargs)
