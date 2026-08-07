"""Prompt bank for the method prompting (L)LMs to diversify text.

All templates are example-based style transfer prompts: style examples are
inserted at ``[STYLE EXAMPLES]`` (and optionally a style name at
``[STYLE NAME]``). Zero-shot templates (prompts without style examples) are
intentionally not supported.
"""

from __future__ import annotations

from diversify_text.method.llm import PLACEHOLDER_TEXT

# -- Prompt key constants -----------------------------------------------------
# Used in both the bank dict and DEFAULT_PROMPT so renaming a key only
# requires a single change.

STYLE_TRANSFER = "style_transfer"
HUMANIZE_TRANSFER = "humanize_transfer"

# Placeholder tokens used inside prompt templates.  The document
# placeholder (``PLACEHOLDER_TEXT``) is shared with other prompt-based
# methods and imported from :mod:`diversify_text.method.llm`.
PLACEHOLDER_STYLE_EXAMPLES = "[STYLE EXAMPLES]"
PLACEHOLDER_STYLE_NAME = "[STYLE NAME]"

#: Placeholders every template must contain.  ``[STYLE NAME]`` is optional.
REQUIRED_PLACEHOLDERS: tuple[str, ...] = (
    PLACEHOLDER_TEXT,
    PLACEHOLDER_STYLE_EXAMPLES,
)

# -- Example-based style transfer templates -----------------------------------
#   These are prompts created by us or inspired by other work that rewrite a
#   text with examples in the target style.
#   These use [STYLE EXAMPLES], [STYLE NAME] and [DOCUMENT SEGMENT] placeholders.

PROMPT_BANK: dict[str, str] = {
    STYLE_TRANSFER: (  # ours
        "Here are examples of the [STYLE NAME] writing style:\n"
        "[STYLE EXAMPLES]\n\n"
        "For the following document give me a diverse paraphrase of the same "
        "that matches the human [STYLE NAME] writing style demonstrated in the examples above. "
        "Preserve the same information. "
        "Output only the paraphrase, nothing else.\n"
        "Document: [DOCUMENT SEGMENT]"
    ),
    HUMANIZE_TRANSFER: (  # inspired by  https://arxiv.org/abs/2401.05952
        "I need to modify a machine-generated text to make it appear more like it was "
        "written by a human. The objective is to introduce elements found in "
        "human-written texts in the [STYLE NAME] style. Here are some examples of "
        "human written texts in [STYLE NAME] style:\n"
        "[STYLE EXAMPLES]\n\n"
        "Please select any combination of these modifications to enhance the text's "
        "human-like quality. The aim is to simulate the imperfections and stylistic "
        "choices typical in the exemplified [STYLE NAME] human style.\n"
        "The word count of the new text should not exceed 1.1 times that of the original "
        "text.\n"
        "You should just give me the revised version without any other words.\n"
        "Here is the machine-generated text: [DOCUMENT SEGMENT]"
    ),
}

#: Template used when no ``prompt`` is selected.
DEFAULT_PROMPT: str = STYLE_TRANSFER
