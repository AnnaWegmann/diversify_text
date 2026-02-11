"""
TinyStyler wrapper module for text style transfer.

Provides a clean interface around the TinyStyler model
(https://huggingface.co/tinystyler/tinystyler) for batched
style-transfer generation.
"""

from __future__ import annotations

import importlib
from typing import Union

import torch
from huggingface_hub import hf_hub_download

StyleInput = Union[torch.Tensor, list[str]]


class TinyStyler:
    """Manages TinyStyler model lifecycle and exposes style-transfer helpers.

    Example
    -------
    >>> ts = TinyStyler()                       # loads model (auto device)
    >>> ts.transfer(["Hello world."],
    ...             style=["Greetings, esteemed world."])
    ['...']
    """

    def __init__(self, device: str | None = None) -> None:
        """Load the TinyStyler model and tokenizer.

        Parameters
        ----------
        device : str, optional
            Torch device string (e.g. ``"cuda"``, ``"cpu"``).
            Auto-detected when *None*.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer, self._model, self._get_style_embeddings_fn = (
            self._load_model()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_style_embedding(self, example_texts: list[str]) -> torch.Tensor:
        """Compute a style embedding from a list of example texts.

        Parameters
        ----------
        example_texts : list[str]
            One or more texts that exemplify the desired style.

        Returns
        -------
        torch.Tensor
            The style embedding tensor.
        """
        return self._get_style_embeddings_fn([example_texts], self.device).to(
            self.device
        )

    def interpolate_style(
        self,
        source: StyleInput,
        target: StyleInput,
        factor: float,
    ) -> torch.Tensor:
        """Linearly interpolate between two styles.

        Parameters
        ----------
        source : torch.Tensor | list[str]
            Source style — a pre-computed embedding **or** example texts.
        target : torch.Tensor | list[str]
            Target style — a pre-computed embedding **or** example texts.
        factor : float
            Interpolation weight in ``[0, 1]``.
            ``0.0`` → pure source, ``1.0`` → pure target.

        Returns
        -------
        torch.Tensor
            Interpolated style embedding.
        """
        source_emb = self._resolve_style(source)
        target_emb = self._resolve_style(target)
        return (1 - factor) * source_emb + factor * target_emb

    def transfer(
        self,
        texts: list[str],
        style: StyleInput,
        *,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> list[str]:
        """Style-transfer a batch of texts.

        Parameters
        ----------
        texts : list[str]
            Input texts to paraphrase / style-transfer.
        style : torch.Tensor | list[str]
            Either a pre-computed style embedding tensor **or** a list of
            example texts in the desired style (the embedding is computed
            automatically).
        max_new_tokens : int
            Maximum number of tokens to generate per text.
        temperature : float
            Sampling temperature.
        top_p : float
            Nucleus-sampling probability mass.

        Returns
        -------
        list[str]
            Style-transferred texts, same length as *texts*.
        """
        style_emb = self._resolve_style(style)

        inputs = self._tokenizer(
            texts, padding="longest", truncation=True, return_tensors="pt"
        ).to(self.device)

        # Expand style embedding to match batch size
        batch_size = inputs["input_ids"].shape[0]
        style_batch = style_emb.expand(batch_size, -1)

        outputs = self._model.generate(
            **inputs,
            style=style_batch,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

        return self._tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_style(self, style: StyleInput) -> torch.Tensor:
        """Return a style embedding tensor, computing it if needed."""
        if isinstance(style, torch.Tensor):
            return style.to(self.device)
        return self.get_style_embedding(style)

    def _load_model(self):
        """Download and initialise the TinyStyler model from HuggingFace."""
        tinystyler_module = importlib.util.module_from_spec(
            importlib.util.spec_from_file_location(
                "tinystyler_hf",
                hf_hub_download(
                    repo_id="tinystyler/tinystyler", filename="tinystyler.py"
                ),
            )
        )
        tinystyler_module.__spec__.loader.exec_module(tinystyler_module)

        tokenizer, model = tinystyler_module.get_tinystyler_model(self.device)
        get_target_style_embeddings = tinystyler_module.get_target_style_embeddings

        return tokenizer, model, get_target_style_embeddings


# ------------------------------------------------------------------
# Module-level convenience function
# ------------------------------------------------------------------


def style_transfer(
    texts: list[str],
    style: StyleInput,
    *,
    device: str | None = None,
    **kwargs,
) -> list[str]:
    """One-shot convenience: load model, transfer, return results.

    Parameters
    ----------
    texts : list[str]
        Input texts.
    style : torch.Tensor | list[str]
        Style embedding or example texts.
    device : str, optional
        Torch device.
    **kwargs
        Forwarded to :meth:`TinyStyler.transfer`
        (``max_new_tokens``, ``temperature``, ``top_p``).

    Returns
    -------
    list[str]
        Style-transferred texts.
    """
    ts = TinyStyler(device=device)
    return ts.transfer(texts, style, **kwargs)
