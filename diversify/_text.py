"""Punctuation-based text splitting for diversify."""

from __future__ import annotations

import re


def split_text_on_punctuation(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;:])\s+", text.strip())
    cleaned = [part.strip() for part in parts if part and part.strip()]
    return cleaned or [text.strip()]
