"""Prompting-based diversification method."""

from __future__ import annotations

import logging
# import re  # TODO: decide what to do with thinking models
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

_DEFAULT_MODEL = "HuggingFaceTB/SmolLM3-3B"
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_TOP_P = 0.9
_MAX_NEW_TOKENS_FACTOR = 2.0
_MAX_NEW_TOKENS_FLOOR = 10
_MAX_NEW_TOKENS_CAP = 2048
_FINEPHRASE_BONUS_TOKENS = 50
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
    ) -> list[tuple[str, str]]:
        """Resolve prompt configuration into an ordered list of (key, template) pairs.

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
        list[tuple[str, str]]
            ``(key, template)`` pairs to cycle through during generation.
            The key identifies the prompt (e.g. ``"finephrase_faq"``);
            the template is the actual prompt string.

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
            templates = [(k, bank[k]) for k in prompt_keys]

        # Case 2: Custom bank without keys → use all its templates.
        elif prompt_bank is not None:
            templates = list(prompt_bank.items())

        # Case 3: Style info without prompt selection → few-shot default.
        elif has_styles:
            templates = [("style_transfer", bank["style_transfer"])]

        # Case 4: No configuration at all → built-in defaults.
        else:
            templates = [(k, bank[k]) for k in DEFAULT_PROMPTS]

        # --- Validate style compatibility ---
        # If styles were provided, the templates must support them
        # via [STYLE EXAMPLES] (example-based) or [STYLE NAME] (name-based).
        if has_styles and not any(
            PLACEHOLDER_STYLE_EXAMPLES in t or PLACEHOLDER_STYLE_NAME in t
            for _k, t in templates
        ):
            raise ValueError(
                "style_example_keys or custom_style_bank were provided, but the "
                "selected prompt template(s) do not contain the "
                f"{PLACEHOLDER_STYLE_EXAMPLES} or {PLACEHOLDER_STYLE_NAME} "
                f"placeholder. Use a style-aware template "
                f"(e.g. prompt_keys=['style_transfer'] or prompt_keys=['reif']) "
                f"or remove style_example_keys. See "
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
        self,
        texts: list[str],
        n: int,
        prompt_templates: list[tuple[str, str]],
        max_new_tokens: int | None,
    ) -> list[int]:
        """Compute per-prompt max_new_tokens for the full n × texts grid.

        When *max_new_tokens* is explicit, every prompt gets that value.
        Otherwise each text gets a budget scaled from its own token count
        (factor ``_MAX_NEW_TOKENS_FACTOR``, clamped between
        ``_MAX_NEW_TOKENS_FLOOR`` and ``_MAX_NEW_TOKENS_CAP``).
        Finephrase templates (keys starting with ``"finephrase_"``) get
        an additional ``_FINEPHRASE_BONUS_TOKENS``.

        Returns a flat list of length ``n * len(texts)`` in the same
        order as the prompt loop: for each style index *i*, for each
        text *j*.
        """
        total = n * len(texts)
        if max_new_tokens is not None:
            return [max_new_tokens] * total

        # Tokenize once to get per-text token counts.
        model = self._ensure_model()
        token_counts = [
            len(ids)
            for ids in model._tokenizer(texts, truncation=True)["input_ids"]
        ]
        # TODO: decide what to do with thinking models
        # is_thinking = getattr(model, "_is_thinking_model", False) is True

        # Build flat list matching the prompt loop order.
        result: list[int] = []
        for i in range(n):
            key, _template = prompt_templates[i % len(prompt_templates)]
            needs_bonus = key.startswith("finephrase_") or "complex" in key
            for count in token_counts:
                budget = max(
                    _MAX_NEW_TOKENS_FLOOR,
                    min(int(count * _MAX_NEW_TOKENS_FACTOR), _MAX_NEW_TOKENS_CAP),
                )
                if needs_bonus:
                    budget = min(budget + _FINEPHRASE_BONUS_TOKENS, _MAX_NEW_TOKENS_CAP)
                # TODO: decide what to do with thinking models
                # if is_thinking:
                #     budget = _MAX_NEW_TOKENS_CAP
                result.append(budget)
        return result

    @staticmethod
    def _resolve_few_shot_examples(**kwargs: Any) -> dict[str, list[str]]:
        """Resolve few-shot style examples from kwargs.

        Returns a dict mapping style names to example sentences, or
        an empty dict when no style kwargs are provided.
        """
        style_keys = kwargs.get("style_example_keys")
        custom_bank = kwargs.get("custom_style_bank")
        if style_keys is not None or custom_bank is not None:
            return resolve_style_sets(custom_bank, style_keys)
        return {}

    def _fill_template(
        self,
        template: str,
        text: str,
        style_idx: int | None = None,
        fs_style_examples: dict[str, list[str]] | None = None,
        n_style_examples: int = _DEFAULT_N_STYLE_EXAMPLES,
    ) -> str:
        """Replace all placeholders in a template to produce a ready prompt.

        Parameters
        ----------
        template : str
            Prompt template containing ``[DOCUMENT SEGMENT]`` and
            optionally ``[STYLE EXAMPLES]`` / ``[STYLE NAME]``.
        text : str
            The input text to insert at ``[DOCUMENT SEGMENT]``.
        style_idx : int or None
            Index into *fs_style_examples* (cycled with modulo).  Each
            paraphrase iteration uses a different style set.  When
            there are fewer styles than paraphrases, styles are reused.
            Required when *fs_style_examples* has more than one entry.
        fs_style_examples : dict[str, list[str]] or None
            Mapping of style names to example sentences.  ``None`` for
            zero-shot templates.
        n_style_examples : int
            Maximum number of example sentences to include from the
            selected style sets.

        Returns
        -------
        str
            Fully filled prompt string, ready for generation.

        Raises
        ------
        ValueError
            If *fs_style_examples* has more than one entry but
            *style_idx* is not provided.
        """
        # Style example placeholders (example-based templates).
        if PLACEHOLDER_STYLE_EXAMPLES in template and fs_style_examples:
            if style_idx is None and len(fs_style_examples) > 1:
                raise ValueError(
                    "style_idx is required when fs_style_examples "
                    "contains more than one style set."
                )
            style_names = list(fs_style_examples.keys())
            style_sets = list(fs_style_examples.values())
            idx = (style_idx or 0) % len(style_sets)
            style_block = self._format_style_examples(
                style_sets[idx], n=n_style_examples
            )
            template = template.replace(PLACEHOLDER_STYLE_EXAMPLES, style_block)

        # Style name placeholder (example-based and name-based templates).
        if PLACEHOLDER_STYLE_NAME in template and fs_style_examples:
            style_names = list(fs_style_examples.keys())
            idx = (style_idx or 0) % len(style_names)
            name = (
                style_names[idx]
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

        prompt_templates = self._resolve_prompts(
            prompt_bank=kwargs.get("prompt_bank"),
            prompt_keys=kwargs.get("prompt_keys"),
            style_example_keys=kwargs.get("style_example_keys"),
            custom_style_bank=kwargs.get("custom_style_bank"),
        )
        all_max_new_tokens = self._compute_max_new_tokens(
            texts, n, prompt_templates, max_new_tokens,
        )
        logger.info(
            "Using %d prompt template(s) for %d paraphrase(s).",
            len(prompt_templates),
            n,
        )

        fs_style_examples = self._resolve_few_shot_examples(**kwargs)
        if fs_style_examples:
            logger.info("Style sets: %s", ", ".join(fs_style_examples.keys()))

        n_ex = kwargs.get("n_style_examples", _DEFAULT_N_STYLE_EXAMPLES)

        # Build prompts in the same order as all_max_new_tokens.
        # TODO: accept texts as an Iterable (not just list) to support
        #       streaming from large files without materialising everything
        #       in memory.
        all_prompts: list[str] = []
        for i in range(n):
            _key, template = prompt_templates[i % len(prompt_templates)]
            for t in texts:
                all_prompts.append(
                    self._fill_template(
                        template=template,
                        text=t,
                        style_idx=i,
                        fs_style_examples=fs_style_examples,
                        n_style_examples=n_ex,
                    )
                )

        # Generate one prompt at a time (each with its own max_new_tokens).
        flat_results = model.generate_text(
            all_prompts,
            max_new_tokens_per_prompt=all_max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # Reshape flat results back into list[list[str]] (len(texts) x n).
        num_texts = len(texts)
        paraphrases_per_text: list[list[str]] = [[] for _ in texts]
        for i in range(n):
            for row_idx in range(num_texts):
                generated = flat_results[i * num_texts + row_idx]
                # TODO: decide what to do with thinking models
                # generated = re.sub(r"<think>.*?</think>\s*", "", generated, flags=re.DOTALL)
                paraphrases_per_text[row_idx].append(generated.strip())

        return paraphrases_per_text
