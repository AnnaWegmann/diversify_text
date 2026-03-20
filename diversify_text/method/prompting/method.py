"""Prompting-based diversification method."""

from __future__ import annotations

import logging
from typing import Any

from diversify_text.method.base import DiversificationMethod
from diversify_text.method.prompting.model import PromptingModel
from diversify_text.method.prompting.prompts import (
    DEFAULT_PROMPT_BANK,
    DEFAULT_PROMPTS,
    FEW_SHOT_PROMPT_BANK,
    PLACEHOLDER_STYLE,
    PLACEHOLDER_STYLE_NAME,
    PLACEHOLDER_TEXT,
)
from diversify_text.styles import resolve_style_sets

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_TOP_P = 0.9
_MAX_NEW_TOKENS_FACTOR = 2.0
_MAX_NEW_TOKENS_FLOOR = 10
_MAX_NEW_TOKENS_CAP = 2048
_DEFAULT_N_STYLE_EXAMPLES = 16


class PromptingMethod(DiversificationMethod):
    """Diversification method that prompts a causal LM for paraphrases."""

    name = "prompting"

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        device: str | None = None,
        precision: str | None = "auto",
    ) -> None:
        """Initialise the prompting method.

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
        """Lazily initialise the model on first use."""
        if self._model is None:
            self._model = PromptingModel(
                model_id=self.model_id, device=self.device, precision=self.precision
            )
            self._model.load()
        return self._model

    @staticmethod
    def _resolve_prompts(
        prompt_bank: dict[str, str] | None = None,
        prompts: list[str] | None = None,
    ) -> list[str]:
        """Resolve prompt configuration into an ordered list of templates.

        Parameters
        ----------
        prompt_bank : dict or None
            Custom prompt bank mapping keys to template strings.
            ``None`` falls back to :data:`DEFAULT_PROMPT_BANK`.
        prompts : list[str] or None
            Select only these keys from the bank.  Order is preserved.

        Returns
        -------
        list[str]
            Prompt template strings to cycle through during generation.
        """
        bank = prompt_bank if prompt_bank is not None else DEFAULT_PROMPT_BANK

        if prompts is not None:
            unknown = set(prompts) - set(bank.keys())
            if unknown:
                raise ValueError(
                    f"Unknown prompt key(s): {sorted(unknown)}. "
                    f"Available: {sorted(bank.keys())}"
                )
            return [bank[k] for k in prompts]

        if prompt_bank is not None:
            return list(prompt_bank.values())

        return [bank[k] for k in DEFAULT_PROMPTS]

    @staticmethod
    def _format_style_examples(
        examples: list[str],
        n: int = _DEFAULT_N_STYLE_EXAMPLES,
    ) -> str:
        """Format a style set into a bullet list for the prompt.

        Parameters
        ----------
        examples : list[str]
            Example sentences from a style set.
        n : int
            Maximum number of examples to include.

        Returns
        -------
        str
            Formatted string with one example per line.
        """
        selected = examples[:n]
        return "\n".join(f'- "{ex}"' for ex in selected)

    def _compute_max_new_tokens(
        self, texts: list[str], max_new_tokens: int | None
    ) -> int:
        """Determine the generation length cap.

        When *max_new_tokens* is ``None``, scales with the longest input
        (factor ``_MAX_NEW_TOKENS_FACTOR``), clamped between
        ``_MAX_NEW_TOKENS_FLOOR`` and ``_MAX_NEW_TOKENS_CAP``.
        An explicit value is returned as-is.
        """
        if max_new_tokens is not None:
            return max_new_tokens
        model = self._ensure_model()
        input_token_counts = [
            len(ids)
            for ids in model._tokenizer(texts, truncation=True)["input_ids"]
        ]
        return max(
            _MAX_NEW_TOKENS_FLOOR,
            min(
                int(max(input_token_counts) * _MAX_NEW_TOKENS_FACTOR),
                _MAX_NEW_TOKENS_CAP,
            ),
        )

    def _resolve_template(self, **kwargs: Any) -> list[str]:
        """Resolve which prompt templates to use.

        When styles are provided but no explicit prompts, defaults to
        the few-shot style transfer prompt from
        :data:`FEW_SHOT_PROMPT_BANK`.

        Returns
        -------
        list[str]
            Prompt template strings to cycle through during generation.
        """
        has_styles = (
            kwargs.get("styles") is not None
            or kwargs.get("style_bank") is not None
        )
        if has_styles and kwargs.get("prompts") is None and kwargs.get("prompt_bank") is None:
            few_shot_key = kwargs.get("few_shot_prompt", "style_transfer")
            return [FEW_SHOT_PROMPT_BANK[few_shot_key]]  # For now only one few shot prompt (pending testing)

        return self._resolve_prompts(
            kwargs.get("prompt_bank"),
            kwargs.get("prompts"),
        )

    @staticmethod
    def _resolve_styles(
        prompt_templates: list[str], **kwargs: Any
    ) -> tuple[list[str], list[list[str]]]:
        """Resolve style sets when the templates need them.

        Returns
        -------
        tuple[list[str], list[list[str]]]
            ``(style_keys, style_sets)``.  Both are empty when the
            templates do not use style placeholders.
        """
        if not any(PLACEHOLDER_STYLE in t for t in prompt_templates):
            return [], []

        return resolve_style_sets(
            kwargs.get("style_bank"),
            kwargs.get("styles"),
        )

    def _fill_template(
        self,
        template: str,
        style_idx: int,
        style_keys: list[str],
        style_sets: list[list[str]],
        n_style_examples: int,
    ) -> str:
        """Fill style placeholders in a template.

        Parameters
        ----------
        template : str
            Prompt template, possibly containing ``[STYLE EXAMPLES]``
            and ``[STYLE NAME]`` placeholders.
        style_idx : int
            Index into *style_sets* (cycled with modulo).
        style_keys, style_sets : list
            Style labels and example lists from :func:`resolve_style_sets`.
        n_style_examples : int
            Number of examples to include from the style set.

        Returns
        -------
        str
            Template with style placeholders filled in.
            ``[DOCUMENT SEGMENT]`` is **not** replaced here.
        """
        if PLACEHOLDER_STYLE not in template or not style_sets:
            return template

        idx = style_idx % len(style_sets)
        style_block = self._format_style_examples(
            style_sets[idx], n=n_style_examples
        )
        template = template.replace(PLACEHOLDER_STYLE, style_block)
        if PLACEHOLDER_STYLE_NAME in template:
            name = (
                style_keys[idx]
                .replace("_tinystyler", "")
                .replace("_stel", "")
                .replace("_", " ")
            )
            template = template.replace(PLACEHOLDER_STYLE_NAME, name)
        return template

    def generate(
        self,
        texts: list[str],
        *,
        n: int,
        max_new_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        **kwargs: Any,
    ) -> list[list[str]]:
        """Generate ``n`` paraphrases for each text.

        Parameters
        ----------
        texts : list[str]
            Input texts to paraphrase.
        n : int
            Number of paraphrases to produce per text.
        max_new_tokens : int or None
            Maximum tokens to generate. ``None`` auto-scales from
            input length (capped at ``_MAX_NEW_TOKENS_CAP``).
        temperature, top_p : float or None
            Sampling parameters. ``None`` uses defaults.
        **kwargs
            Extra options forwarded from ``Diversifier``, including
            ``prompts``, ``prompt_bank``, ``styles``, ``style_bank``,
            ``few_shot_prompt``, and ``n_style_examples``.
        """
        model = self._ensure_model()
        temperature = temperature if temperature is not None else _DEFAULT_TEMPERATURE
        top_p = top_p if top_p is not None else _DEFAULT_TOP_P
        max_new_tokens = self._compute_max_new_tokens(texts, max_new_tokens)

        prompt_templates = self._resolve_template(**kwargs)
        style_keys, style_sets = self._resolve_styles(prompt_templates, **kwargs)
        n_ex = kwargs.get("n_style_examples", _DEFAULT_N_STYLE_EXAMPLES)

        logger.info(
            "Using %d prompt template(s) for %d paraphrase(s).",
            len(prompt_templates),
            n,
        )
        if style_keys:
            logger.info("Style sets: %s", ", ".join(style_keys))

        paraphrases_per_text: list[list[str]] = [[] for _ in texts]

        for i in range(n):
            template = prompt_templates[i % len(prompt_templates)]
            template = self._fill_template(
                template, i, style_keys, style_sets, n_ex
            )
            filled = [template.replace(PLACEHOLDER_TEXT, t) for t in texts]
            logger.debug("Prompt (iteration %d):\n%s", i, filled[0])

            batch = model.generate_text(
                filled,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            for row_idx, generated in enumerate(batch):
                paraphrases_per_text[row_idx].append(generated.strip())

        return paraphrases_per_text
