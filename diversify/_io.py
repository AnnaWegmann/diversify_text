"""Input normalisation and tabular file loading for diversify."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

TextInput = Union[str, list[str], pd.Series, pd.DataFrame]


def normalize_input(texts: TextInput, text_column: str) -> list[str]:
    """Coerce any supported input type into ``list[str]``."""
    if isinstance(texts, str):
        return [texts]
    if isinstance(texts, pd.Series):
        return texts.tolist()
    if isinstance(texts, pd.DataFrame):
        return texts[text_column].tolist()
    if isinstance(texts, list):
        return texts
    raise TypeError(
        f"Unsupported input type {type(texts).__name__}. "
        "Expected str, list[str], pd.Series, or pd.DataFrame."
    )


def load_tabular_input(
    texts: TextInput, text_column: str
) -> tuple[pd.DataFrame, Path, str] | None:
    """Load CSV/TSV input when *texts* points to a supported file path.

    Returns ``None`` for any non-file input.
    """
    if not isinstance(texts, str):
        return None
    path = Path(texts)
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".tsv"} or not path.is_file():
        return None
    sep = "," if suffix == ".csv" else "\t"
    df = pd.read_csv(path, sep=sep)
    if text_column not in df.columns:
        available = ", ".join(df.columns)
        raise ValueError(
            f"Column '{text_column}' not found in {path}. Available: {available}"
        )
    df[text_column] = df[text_column].fillna("").astype(str).tolist()
    return df, path, sep
