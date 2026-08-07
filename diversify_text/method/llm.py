"""Shared causal language model for prompt-based methods.

:class:`PromptingModel` wraps the model itself (load + generate);
:class:`CausalLMMethod` is the shared base class for methods that
generate through it.  Uses the ``transformers`` library
(``AutoModelForCausalLM``).

.. note:: vLLM support is planned for a future release.
"""

from __future__ import annotations

import logging

import torch
from huggingface_hub import snapshot_download

from diversify_text._cache import model_cache
from diversify_text._utils import default_device, spinner, suppress_hf_load_noise
from diversify_text.method.base import DiversificationMethod

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "HuggingFaceTB/SmolLM3-3B"


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
        # TODO: add vLLM backend support in a future release.
        self._load_transformers()

    def generate_text(
        self,
        prompts: list[str],
        *,
        max_new_tokens_per_prompt: list[int],
        temperature: float,
        top_p: float,
    ) -> list[str]:
        """Generate completions for a list of prompts.

        Each prompt is processed individually with its own
        ``max_new_tokens`` value.

        Parameters
        ----------
        prompts : list[str]
            Prompts to generate completions for.
        max_new_tokens_per_prompt : list[int]
            Maximum tokens to generate for each prompt (same length
            as *prompts*).
        temperature, top_p
            Sampling parameters forwarded to the backend.
        """
        # TODO: add batching and vLLM backend support in a future release.
        results: list[str] = []
        for prompt, max_new_tokens in zip(prompts, max_new_tokens_per_prompt):
            results.extend(
                self._generate_transformers(
                    [prompt],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            )
        return results

    def _load_transformers(self) -> None:
        """Download weights (if needed) and load the model into memory."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        with spinner(f"Downloading {self.model_id}"):
            snapshot_download(self.model_id)
        with spinner(f"Loading {self.model_id} (transformers, {self.device})"):
            with suppress_hf_load_noise():
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                self._tokenizer.padding_side = "left"  # required for decoder-only models
                load_kwargs: dict = {}
                if self._torch_dtype is not None:
                    load_kwargs["torch_dtype"] = self._torch_dtype
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, **load_kwargs
                )
                self._model.to(self.device)
                self._model.eval()
        # TODO: decide what to do with thinking models
        self._is_thinking_model = (
            hasattr(self._tokenizer, "chat_template")
            and self._tokenizer.chat_template is not None
            and "<think>" in self._tokenizer.chat_template
        )

    # -- vLLM backend ---------------------------------------------------------
    # TBD

    # -- transformers backend -------------------------------------------------

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
            # TODO: decide what to do with thinking models
            kwargs = {}
            if self._is_thinking_model:
                kwargs["enable_thinking"] = False
            return self._tokenizer.apply_chat_template(
                conversations,
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )
        return prompts


@model_cache
def _load_llm(
    model_id: str,
    device: str,
    precision: str | None,
) -> PromptingModel:
    """Load a causal language model (cached, shared across methods)."""
    llm = PromptingModel(model_id=model_id, device=device, precision=precision)
    llm.load()
    return llm


class CausalLMMethod(DiversificationMethod):
    """Shared base for methods that generate via a causal language model.

    Handles the model configuration and lazy loading; subclasses
    implement :meth:`generate`.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        device: str | None = None,
        precision: str | None = "auto",
    ) -> None:
        """Initialise the method.

        Parameters
        ----------
        model : str
            HuggingFace model identifier.
        device : str or None
            Torch device (e.g. ``"cpu"``, ``"mps"``, ``"cuda"``).
            Defaults to auto-detection via :func:`default_device`.
        precision : str or None
            Model weight precision: ``"auto"`` (bfloat16), ``"float16"``,
            ``"bfloat16"``, or ``None`` (float32).
        """
        self.model_id = model
        self.device = device
        self.precision = precision
        self._model: PromptingModel | None = None

    def prepare(self) -> None:
        """Download and load the underlying language model."""
        self._ensure_model()

    def _ensure_model(self) -> PromptingModel:
        """Fetch the shared model on first use."""
        if self._model is None:
            self._model = _load_llm(
                self.model_id, self.device or default_device(), self.precision
            )
        return self._model
