"""Shared bios loader for example scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_bios(path: str | Path | None = None, text_column: str = "bio") -> pd.DataFrame:
    """Load bios CSV and normalize the text column."""
    if path is None:
        path = Path(__file__).resolve().parents[1] / "data" / "bios_400.csv"
    df = pd.read_csv(path)
    if text_column not in df.columns:
        available = ", ".join(df.columns)
        raise ValueError(
            f"Column '{text_column}' not found in {path}. Available: {available}"
        )
    df[text_column] = df[text_column].fillna("").astype(str).str.strip()
    return df


if __name__ == "__main__":
    bios = load_bios()
    print(f"Loaded {len(bios)} bios")
    print(bios.head())
