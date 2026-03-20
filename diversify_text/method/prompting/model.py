"""Model wrapper for prompt-based text generation.

Uses the ``transformers`` library (``AutoModelForCausalLM``), which is
already a project dependency.

.. note:: More efficient inference backends such as vLLM (or datatrove wrapper) are
   planned to be considered for future releases.
"""

from __future__ import annotations

import logging

import torch
from huggingface_hub import snapshot_download

from diversify_text._utils import default_device, spinner, suppress_hf_load_noise

logger = logging.getLogger(__name__)


class PromptingModel:
    """Manages a causal LM for prompt-based generation."""

    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        precision: str | None = "auto",
    ) -> None:
        """Initialise the model wrapper.

        Parameters
        ----------
        model_id : str
            HuggingFace model identifier.
        device : str or None
            Torch device. Defaults to auto-detection.
        precision : str or None
            Weight precision: ``"auto"`` (bfloat16), ``"float16"``,
            ``"bfloat16"``, or ``None`` (float32).
        """
        self.model_id = model_id
        self.device = device or default_device()
        self.precision = precision
        self._torch_dtype = self._resolve_dtype()
        self._tokenizer = None
        self._model = None

    def _resolve_dtype(self) -> torch.dtype | None:
        """Map *precision* string to a ``torch.dtype``.

        Returns
        -------
        torch.dtype or None
            The dtype to pass to ``from_pretrained``, or ``None``
            for full precision (float32).
        """
        if self.precision == "auto":
            return torch.bfloat16
        if self.precision is None:
            return None  # full precision (float32)
        mapping: dict[str, torch.dtype] = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if self.precision not in mapping:
            raise ValueError(
                f"Unknown precision {self.precision!r}. "
                f"Choose from: 'auto', 'float16', 'bfloat16', or None (float32)"
            )
        return mapping[self.precision]

    def load(self) -> None:
        """Download and load the model."""
        self._load_transformers()  # might use a different backend in the future

    def generate_text(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[str]:
        """Generate completions for a batch of prompts."""
        return self._generate_transformers(  # might use a different backend in the future
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def _load_transformers(self) -> None:
        """Download weights (if needed) and load the model into memory."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        with spinner(f"Downloading {self.model_id}"):
            snapshot_download(self.model_id)
        with spinner(f"Loading {self.model_id} (transformers, {self.device})"):
            with suppress_hf_load_noise():
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                if self._tokenizer.pad_token is None: # no pad token for smolLM
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                load_kwargs: dict = {}
                if self._torch_dtype is not None:
                    load_kwargs["torch_dtype"] = self._torch_dtype
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, **load_kwargs
                )
                self._model.to(self.device)
                self._model.eval()

    def _generate_transformers(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[str]:
        """Run batched generation via the transformers ``generate()`` API."""
        formatted = self._apply_chat_template(prompts)
        inputs = self._tokenizer(
            formatted, padding=True, return_tensors="pt", truncation=True
        ).to(self.device)
        input_len = inputs["input_ids"].shape[1]

        max_context = getattr(self._model.config, "max_position_embeddings", None)
        if max_context is None:
            raise ValueError(
                f"Model {self.model_id!r} does not expose "
                f"'max_position_embeddings' in its config. "
                f"Cannot determine context window size."
            )
        if input_len > max_context:
            logger.warning(
                "Input length (%d tokens) exceeds model context window (%d). "
                "Output quality may degrade.",
                input_len,
                max_context,
            )

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = output_ids[:, input_len:]
        return self._tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    def _apply_chat_template(self, prompts: list[str]) -> list[str]:
        """Format prompts for instruct models when a chat template exists."""
        if (
            hasattr(self._tokenizer, "chat_template")
            and self._tokenizer.chat_template is not None
        ):
            conversations = [[{"role": "user", "content": p}] for p in prompts]
            return self._tokenizer.apply_chat_template(
                conversations,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompts
