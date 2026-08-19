"""Zero-shot diversification method: styles defined by rewrite instructions."""

from __future__ import annotations

import logging
from typing import Any

from diversify_text.method.llm import CausalLMMethod, PLACEHOLDER_TEXT
from diversify_text.method.zero_shot.bank import ZERO_SHOT_STYLE_BANK

logger = logging.getLogger(__name__)

#: Appended to instructions that do not place the input text themselves.
_SCAFFOLD = "\n\nOutput only the rewrite, nothing else.\nText: "


class ZeroShotMethod(CausalLMMethod):
    """Diversification method whose styles are defined by instructions.

    Instead of example texts, each style is one rewrite instruction for
    the causal language model.  An instruction may place the input text
    itself via the ``[DOCUMENT SEGMENT]`` placeholder; otherwise the
    text is appended at the end with a fixed scaffold.
    """

    name = "zero_shot"
    style_bank = ZERO_SHOT_STYLE_BANK
    # The default unusual and surface styles are example-based
    unusual_style_bank: dict[str, list[str]] = {}
    surface_style_bank: dict[str, list[str]] = {}

    def generate(
        self,
        texts: list[str],
        style_dict: dict[str, list[str]],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs: Any,
    ) -> list[list[str]]:
        """Generate one rewrite per style in *style_dict* for each text.

        Parameters
        ----------
        texts : list[str]
            Input texts to rewrite.
        style_dict : dict[str, list[str]]
            Maps each target style name to exactly one rewrite
            instruction (a longer list raises).
        max_new_tokens, temperature, top_p
            Generation parameters; ``None`` uses defaults.

        Returns
        -------
        list[list[str]]
            For each input text, one generated string per style, in
            *style_dict* order.
        """
        for name, instructions in style_dict.items():
            if len(instructions) != 1:
                raise ValueError(
                    f"Style {name!r} must have exactly one instruction, "
                    f"got {len(instructions)}."
                )
        logger.info("Using instruction style(s): %s", ", ".join(style_dict))

        all_prompts: list[str] = []
        for instructions in style_dict.values():
            instruction = instructions[0]
            for text in texts:
                all_prompts.append(self._fill_instruction(instruction, text))

        return self._run_prompts(
            all_prompts,
            texts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    @staticmethod
    def _fill_instruction(instruction: str, text: str) -> str:
        """Fill the input text into an instruction.

        The instruction may place the text itself via the
        ``[DOCUMENT SEGMENT]`` placeholder; otherwise the text is
        appended at the end with a fixed scaffold.
        """
        if PLACEHOLDER_TEXT in instruction:
            return instruction.replace(PLACEHOLDER_TEXT, text)
        return f"{instruction}{_SCAFFOLD}{text}"
