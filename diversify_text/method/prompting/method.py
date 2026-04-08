"""Prompting-based diversification method."""

from __future__ import annotations

import logging
from typing import Any

from diversify_text.method.base import DiversificationMethod
from diversify_text.method.prompting.model import PromptingModel
from diversify_text.method.prompting.prompts import (
    DEFAULT_PROMPTS,
    PLACEHOLDER_STYLE_EXAMPLES,
    PLACEHOLDER_STYLE_NAME,
    PLACEHOLDER_TEXT,
    PROMPT_BANK,
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
        prompt_keys: list[str] | None = None,
        style_example_keys: list[str] | None = None,
        custom_style_bank: dict[str, list[str]] | None = None,
    ) -> list[str]:
        """Resolve prompt configuration into an ordered list of templates.

        Selects from a unified bank that contains both zero-shot and
        few-shot templates.  The distinction is implicit: few-shot
        templates contain ``[STYLE EXAMPLES]`` placeholders, zero-shot
        templates do not.

        Parameters
        ----------
        prompt_bank : dict or None
            Custom prompt bank mapping keys to template strings.
            ``None`` falls back to :data:`PROMPT_BANK` (the merged
            zero-shot + few-shot bank).
        prompt_keys : list[str] or None
            Select only these keys from the bank.  Order is preserved.
        style_example_keys : list[str] or None
            Names of style sets for few-shot examples.  If provided
            without *prompt_keys*, the method automatically selects
            the ``"style_transfer"`` prompt template.  When combined
            with *prompt_keys*, the selected templates must contain
            the ``[STYLE EXAMPLES]`` placeholder.
        custom_style_bank : dict or None
            Custom style bank — same trigger behavior as
            *style_example_keys*.

        Returns
        -------
        list[str]
            Prompt template strings to cycle through during generation.

        Raises
        ------
        ValueError
            If *prompt_keys* contains unknown keys, or if *style_example_keys*
            / *custom_style_bank* are provided but the selected
            templates do not contain a ``[STYLE EXAMPLES]``
            placeholder.
        """
        bank = prompt_bank if prompt_bank is not None else PROMPT_BANK
        has_styles = style_example_keys is not None or custom_style_bank is not None

        # --- Select templates (four mutually exclusive cases) ---

        # Case 1: Explicit keys → pick those templates from the bank.
        if prompt_keys is not None:
            unknown = set(prompt_keys) - set(bank.keys())
            if unknown:
                raise ValueError(
                    f"Unknown prompt key(s): {sorted(unknown)}. "
                    f"Available: {sorted(bank.keys())}"
                )
            templates = [bank[k] for k in prompt_keys]

        # Case 2: Custom bank without keys → use all its templates.
        elif prompt_bank is not None:
            templates = list(prompt_bank.values())

        # Case 3: Style info without prompt selection → few-shot default.
        elif has_styles:
            templates = [bank["style_transfer"]]

        # Case 4: No configuration at all → built-in defaults.
        else:
            templates = [bank[k] for k in DEFAULT_PROMPTS]

        # --- Validate style compatibility ---
        # If styles were provided, the templates must support them.
        if has_styles and not any(PLACEHOLDER_STYLE_EXAMPLES in t for t in templates):
            raise ValueError(
                "style_example_keys or custom_style_bank were provided, but the "
                "selected prompt template(s) do not contain the "
                f"{PLACEHOLDER_STYLE_EXAMPLES} placeholder. Use a few-shot "
                f"template (e.g. prompt_keys=['style_transfer']) or remove "
                f"style_example_keys. See "
                f"https://annawegmann.github.io/diversify_text/prompts.html"
            )

        return templates

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

    @staticmethod
    def _resolve_styles(
        prompt_templates: list[str], **kwargs: Any
    ) -> tuple[list[str], list[list[str]]]:
        """Resolve style sets when the templates need them.

        Returns
        -------
        tuple[list[str], list[list[str]]]
            ``(style_example_keys, style_sets)``.  Both are empty when the
            templates do not use style placeholders.
        """
        if not any(PLACEHOLDER_STYLE_EXAMPLES in t for t in prompt_templates):
            return [], []

        return resolve_style_sets(
            kwargs.get("custom_style_bank"),
            kwargs.get("style_example_keys"),
        )

    def _fill_template(
        self,
        template: str,
        text: str,
        style_idx: int,
        style_example_keys: list[str],
        style_sets: list[list[str]],
        n_style_examples: int,
    ) -> str:
        """Replace all placeholders in a template to produce a ready prompt.

        Parameters
        ----------
        template : str
            Prompt template containing ``[DOCUMENT SEGMENT]`` and
            optionally ``[STYLE EXAMPLES]`` / ``[STYLE NAME]``.
        text : str
            The input text to insert at ``[DOCUMENT SEGMENT]``.
        style_idx : int
            Index into *style_sets* (cycled with modulo).  Each
            paraphrase iteration uses a different style set.  When
            there are fewer styles than paraphrases, styles are reused.
        style_example_keys, style_sets : list
            Style labels and example lists from :func:`resolve_style_sets`.
        n_style_examples : int
            Number of examples to include from the style set.

        Returns
        -------
        str
            Fully filled prompt string, ready for generation.
        """
        # Style placeholders (few-shot only).
        if PLACEHOLDER_STYLE_EXAMPLES in template and style_sets:
            idx = style_idx % len(style_sets)
            style_block = self._format_style_examples(
                style_sets[idx], n=n_style_examples
            )
            template = template.replace(PLACEHOLDER_STYLE_EXAMPLES, style_block)
            if PLACEHOLDER_STYLE_NAME in template:
                name = (
                    style_example_keys[idx]
                    .replace("_tinystyler", "")
                    .replace("_stel", "")
                    .replace("_", " ")
                )
                template = template.replace(PLACEHOLDER_STYLE_NAME, name)

        # Document placeholder (all templates).
        template = template.replace(PLACEHOLDER_TEXT, text)
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
            ``prompt_keys``, ``prompt_bank``, ``style_example_keys``,
            ``custom_style_bank``, and ``n_style_examples``.
        """
        model = self._ensure_model()
        temperature = temperature if temperature is not None else _DEFAULT_TEMPERATURE
        top_p = top_p if top_p is not None else _DEFAULT_TOP_P
        max_new_tokens = self._compute_max_new_tokens(texts, max_new_tokens)

        prompt_templates = self._resolve_prompts(
            prompt_bank=kwargs.get("prompt_bank"),
            prompt_keys=kwargs.get("prompt_keys"),
            style_example_keys=kwargs.get("style_example_keys"),
            custom_style_bank=kwargs.get("custom_style_bank"),
        )
        style_example_keys, style_sets = self._resolve_styles(prompt_templates, **kwargs)
        n_ex = kwargs.get("n_style_examples", _DEFAULT_N_STYLE_EXAMPLES)

        logger.info(
            "Using %d prompt template(s) for %d paraphrase(s).",
            len(prompt_templates),
            n,
        )
        if style_example_keys:
            logger.info("Style sets: %s", ", ".join(style_example_keys))

        # Build all n * len(texts) filled prompts as a flat list.
        # Order: all texts for iteration 0, then all texts for iteration 1, etc.
        # TODO: accept texts as an Iterable (not just list) to support
        #       streaming from large files without materialising everything
        #       in memory.
        all_prompts = [
            self._fill_template(
                template=prompt_templates[i % len(prompt_templates)],
                text=t,
                style_idx=i,
                style_example_keys=style_example_keys,
                style_sets=style_sets,
                n_style_examples=n_ex,
            )
            for i in range(n)
            for t in texts
        ]

        # Single model call — the model chunks internally.
        flat_results = model.generate_text(
            all_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # Reshape flat results back into list[list[str]] (len(texts) x n).
        num_texts = len(texts)
        paraphrases_per_text: list[list[str]] = [[] for _ in texts]
        for i in range(n):
            for row_idx in range(num_texts):
                generated = flat_results[i * num_texts + row_idx]
                # Strip leading/trailing whitespace from model output.
                # This is safe — space variation within the paraphrase is preserved.
                paraphrases_per_text[row_idx].append(generated.strip())

        return paraphrases_per_text
