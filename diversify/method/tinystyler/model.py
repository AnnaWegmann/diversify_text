"""
TinyStyler wrapper module for text style transfer.

Provides a clean interface around the TinyStyler model
(https://huggingface.co/tinystyler/tinystyler) for batched
style-transfer generation.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Union

import torch
from huggingface_hub import hf_hub_download

StyleInput = Union[torch.Tensor, list[str]]


class TinyStyler:
    """Manages TinyStyler model lifecycle and exposes style-transfer helpers."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer, self._model, self._get_style_embeddings_fn = (
            self._load_model()
        )

    def get_style_embedding(self, example_texts: list[str]) -> torch.Tensor:
        return self._get_style_embeddings_fn([example_texts], self.device).to(
            self.device
        )

    def interpolate_style(
        self,
        source: StyleInput,
        target: StyleInput,
        factor: float,
    ) -> torch.Tensor:
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
        style_emb = self._resolve_style(style)

        inputs = self._tokenizer(
            texts, padding="longest", truncation=True, return_tensors="pt"
        ).to(self.device)

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

    def _resolve_style(self, style: StyleInput) -> torch.Tensor:
        if isinstance(style, torch.Tensor):
            return style.to(self.device)
        return self.get_style_embedding(style)

    def _load_model(self):
        tinystyler_module = importlib.util.module_from_spec(
            importlib.util.spec_from_file_location(
                "tinystyler_hf",
                hf_hub_download(
                    repo_id="tinystyler/tinystyler", filename="tinystyler.py"
                ),
            )
        )
        tinystyler_module.__spec__.loader.exec_module(tinystyler_module)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*tie_word_embeddings.*")  # harmless: TinyStyler saves weights explicitly
            tokenizer, model = tinystyler_module.get_tinystyler_model(self.device)
        get_target_style_embeddings = tinystyler_module.get_target_style_embeddings

        return tokenizer, model, get_target_style_embeddings


def style_transfer(
    texts: list[str],
    style: StyleInput,
    *,
    device: str | None = None,
    **kwargs,
) -> list[str]:
    """One-shot convenience: load model, transfer, return results."""
    ts = TinyStyler(device=device)
    return ts.transfer(texts, style, **kwargs)
