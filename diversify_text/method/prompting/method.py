"""Prompting-based diversification method."""

from __future__ import annotations

import logging
# import re  # TODO: decide what to do with thinking models
from typing import Any

from diversify_text.method.llm import CausalLMMethod
from diversify_text.method.prompting.prompts import (
    DEFAULT_PROMPT,
    PLACEHOLDER_STYLE_EXAMPLES,
    PLACEHOLDER_STYLE_NAME,
    PLACEHOLDER_TEXT,
    PROMPT_BANK,
    REQUIRED_PLACEHOLDERS,
)

logger = logging.getLogger(__name__)

_DEFAULT_N_STYLE_EXAMPLES = 16


class PromptingMethod(CausalLMMethod):
    """Diversification method that prompts a causal LM for paraphrases."""

    name = "prompting"

    @staticmethod
    def _resolve_prompt(prompt: str | None = None) -> tuple[str, str]:
        """Resolve the *prompt* option into a single ``(key, template)`` pair.

        Exactly one template is active per call, and *prompt* selects
        which one:

        * ``None`` — use the default built-in template
          (:data:`DEFAULT_PROMPT`).
        * the name of a built-in template from :data:`PROMPT_BANK`
          (e.g. ``"humanize_transfer"``).
        * the full text of your own template (any string that is not a
          built-in name, recognised by containing placeholder tokens).
          It must contain **all** required placeholders
          (``[DOCUMENT SEGMENT]`` and ``[STYLE EXAMPLES]``);
          ``[STYLE NAME]`` is optional.  Unknown bracket tokens are left
          as literal text.

        Returns
        -------
        tuple[str, str]
            The ``(key, template)`` pair; custom templates get the key
            ``"custom"``.

        Raises
        ------
        ValueError
            If a custom template is missing required placeholders (the
            error names every missing one), or if *prompt* is neither a
            known key nor a template.
        """
        if prompt is None:
            return DEFAULT_PROMPT, PROMPT_BANK[DEFAULT_PROMPT]
        if prompt in PROMPT_BANK:
            return prompt, PROMPT_BANK[prompt]
        # A string containing any placeholder is a custom template.
        placeholders = (
            PLACEHOLDER_TEXT,
            PLACEHOLDER_STYLE_EXAMPLES,
            PLACEHOLDER_STYLE_NAME,
        )
        if any(p in prompt for p in placeholders):
            missing = [p for p in REQUIRED_PLACEHOLDERS if p not in prompt]
            if missing:
                raise ValueError(
                    "Custom prompt template is missing required "
                    f"placeholder(s): {', '.join(missing)}. Every template "
                    "must contain both "
                    f"{PLACEHOLDER_TEXT} and {PLACEHOLDER_STYLE_EXAMPLES} "
                    f"({PLACEHOLDER_STYLE_NAME} is optional)."
                )
            return "custom", prompt
        raise ValueError(
            f"Unknown prompt {prompt!r}. Available keys: "
            f"{sorted(PROMPT_BANK)}. To use your own template instead, pass "
            "a string containing the "
            f"{PLACEHOLDER_TEXT} and {PLACEHOLDER_STYLE_EXAMPLES} placeholders."
        )

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
            ``[STYLE EXAMPLES]``, and optionally ``[STYLE NAME]``.
        text : str
            The input text to insert at ``[DOCUMENT SEGMENT]``.
        style_idx : int or None
            Index into *fs_style_examples* (cycled with modulo).  Each
            paraphrase iteration uses a different style set.  When
            there are fewer styles than paraphrases, styles are reused.
            Required when *fs_style_examples* has more than one entry.
        fs_style_examples : dict[str, list[str]] or None
            Mapping of style names to example sentences.
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
        style_dict: dict[str, list[str]],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        prompt: str | None = None,
        n_style_examples: int = _DEFAULT_N_STYLE_EXAMPLES,
        **kwargs: Any,
    ) -> list[list[str]]:
        """Generate one paraphrase per style in *style_dict* for each text.

        Parameters
        ----------
        texts : list[str]
            Input texts to paraphrase.
        style_dict : dict[str, list[str]]
            Maps each target style name to its example texts (as built
            by :func:`~diversify_text.styles.resolve_style_dict`).
        max_new_tokens, temperature, top_p
            As in :meth:`generate`; ``None`` uses defaults.
        prompt : str or None
            Prompt selection, see :meth:`_resolve_prompt`.
        n_style_examples : int
            Maximum number of example texts inserted per prompt.

        Returns
        -------
        list[list[str]]
            For each input text, one generated string per style, in
            *style_dict* order.
        """
        prompt_key, prompt_template = self._resolve_prompt(prompt)
        n = len(style_dict)
        logger.info("Using prompt template %r for %d paraphrase(s).", prompt_key, n)
        logger.info("Style sets: %s", ", ".join(style_dict.keys()))

        # TODO: accept texts as an Iterable (not just list) to support
        #       streaming from large files without materialising everything
        #       in memory.
        # Paraphrase slot i uses style i, always with the single active
        # template.
        all_prompts: list[str] = []
        for style_idx in range(n):
            for t in texts:
                all_prompts.append(
                    self._fill_template(
                        template=prompt_template,
                        text=t,
                        style_idx=style_idx,
                        fs_style_examples=style_dict,
                        n_style_examples=n_style_examples,
                    )
                )

        return self._run_prompts(
            all_prompts,
            texts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
