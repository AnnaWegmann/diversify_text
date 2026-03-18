"""Prompting method submodule."""

from diversify_text.method.prompting.method import PromptingMethod
from diversify_text.method.prompting.prompts import DEFAULT_PROMPT_BANK, DEFAULT_PROMPTS, FEW_SHOT_PROMPT_BANK

__all__ = ["PromptingMethod", "DEFAULT_PROMPT_BANK", "DEFAULT_PROMPTS", "FEW_SHOT_PROMPT_BANK"]
