"""Input resolution for diversify.

Converts the many input forms users can provide (single string, list,
generator, CSV/TSV/TXT file path) into a uniform ``Iterator[str]`` plus
an :class:`InputContext` that describes the source.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Union


# ------------------------------------------------------------------
# Type alias
# ------------------------------------------------------------------

TextInput = Union[str, "list[str]", "Iterable[str]"]

_log = logging.getLogger("diversify")


# ------------------------------------------------------------------
# Input kind & context
# ------------------------------------------------------------------


class InputKind(Enum):
    """Discriminator for how the user provided input."""

    SINGLE_STR = auto()
    LIST = auto()
    ITERABLE = auto()
    FILE_CSV = auto()
    FILE_TSV = auto()
    FILE_TXT = auto()


@dataclass(frozen=True)
class InputContext:
    """Read-only metadata about the resolved input source."""

    kind: InputKind
    input_path: Path | None = None
    text_column: str | None = None
    total: int | None = None


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def resolve_input(
    texts: TextInput,
    text_column: str = "text",
) -> tuple[Iterator[str], InputContext]:
    """Convert any supported input into a lazy ``Iterator[str]`` plus metadata.

    Parameters
    ----------
    texts : str | list[str] | Iterable[str]
        A single text, a list of texts, a generator / iterable of texts,
        or a path to a ``.csv``, ``.tsv``, or ``.txt`` file.
    text_column : str
        Column name to extract when *texts* points to a CSV/TSV file.

    Returns
    -------
    (Iterator[str], InputContext)
    """
    # --- str: could be a file path or a single text ---
    if isinstance(texts, str):
        path = Path(texts)
        suffix = path.suffix.lower()

        if suffix == ".csv" and path.is_file():
            _validate_csv_header(path, text_column, delimiter=",")
            total = _count_file_lines(path) - 1  # subtract header row
            return _iter_csv(path, text_column, delimiter=","), InputContext(
                kind=InputKind.FILE_CSV,
                input_path=path,
                text_column=text_column,
                total=total,
            )

        if suffix == ".tsv" and path.is_file():
            _validate_csv_header(path, text_column, delimiter="\t")
            total = _count_file_lines(path) - 1  # subtract header row
            return _iter_csv(path, text_column, delimiter="\t"), InputContext(
                kind=InputKind.FILE_TSV,
                input_path=path,
                text_column=text_column,
                total=total,
            )

        if suffix == ".txt" and path.is_file():
            total = _count_nonempty_lines(path)
            _log.warning(
                "TXT file input: each line is treated as a separate text "
                "to diversify. Newlines are never part of a parsed text. "
                "(%d non-empty lines found in %s)",
                total,
                path,
            )
            return _iter_txt_lines(path), InputContext(
                kind=InputKind.FILE_TXT,
                input_path=path,
                total=total,
            )

        # Not a recognized file — treat as a single text.
        return iter([texts]), InputContext(kind=InputKind.SINGLE_STR, total=1)

    # --- list[str] ---
    if isinstance(texts, list):
        return iter(texts), InputContext(kind=InputKind.LIST, total=len(texts))

    # --- generic Iterable[str] (generators, file handles, …) ---
    if isinstance(texts, Iterable):
        return iter(texts), InputContext(kind=InputKind.ITERABLE, total=None)

    raise TypeError(
        f"Unsupported input type {type(texts).__name__}. "
        "Expected str, list[str], or Iterable[str]."
    )


# ------------------------------------------------------------------
# Validation & counting (private, cheap first-pass helpers)
# ------------------------------------------------------------------


def _validate_csv_header(path: Path, text_column: str, delimiter: str) -> None:
    """Check that *text_column* exists in the CSV/TSV header.

    Opens the file, reads only the header row, then closes it.

    Parameters
    ----------
    path : Path
        Path to the CSV or TSV file.
    text_column : str
        Expected column name.
    delimiter : str
        Field separator — ``","`` for CSV, ``"\\t"`` for TSV.

    Raises
    ------
    ValueError
        If *text_column* is not found among the file's header fields.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None or text_column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames or [])
            raise ValueError(
                f"Column '{text_column}' not found in {path}. "
                f"Available: {available}"
            )


def _count_file_lines(path: Path) -> int:
    """Count the total number of lines in a file.

    This is a cheap pass that only reads raw lines — no CSV parsing.
    For CSV/TSV files, subtract 1 for the header to get the data-row
    count.  The count may slightly overestimate if the file contains
    multi-line quoted CSV fields, but that is acceptable for a progress
    bar.

    Parameters
    ----------
    path : Path
        Path to the file.

    Returns
    -------
    int
        Total number of lines.
    """
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def _count_nonempty_lines(path: Path) -> int:
    """Count non-empty lines in a file (skips blank / whitespace-only).

    Parameters
    ----------
    path : Path
        Path to the file.

    Returns
    -------
    int
        Number of non-empty lines.
    """
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# ------------------------------------------------------------------
# Lazy file iterators (private)
# ------------------------------------------------------------------


def _iter_csv(path: Path, text_column: str, delimiter: str) -> Iterator[str]:
    """Lazily yield *text_column* values from a CSV/TSV file.

    Only one row is held in memory at a time.  The file handle is
    closed automatically when the generator is exhausted or garbage
    collected.

    Parameters
    ----------
    path : Path
        Path to the CSV or TSV file.
    text_column : str
        Name of the column that contains the texts to diversify.
    delimiter : str
        Field separator — ``","`` for CSV, ``"\\t"`` for TSV.

    Yields
    ------
    str
        The text value from each row.
    """
    f = open(path, newline="", encoding="utf-8")
    try:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            yield row.get(text_column) or ""
    finally:
        f.close()


def _iter_txt_lines(path: Path) -> Iterator[str]:
    """Lazily yield non-empty, stripped lines from a ``.txt`` file.

    Blank lines and whitespace-only lines are skipped.  Only one line
    is held in memory at a time.  The file handle is closed
    automatically when the generator is exhausted or garbage collected.

    Parameters
    ----------
    path : Path
        Path to the ``.txt`` file.

    Yields
    ------
    str
        Each non-empty line, with leading/trailing whitespace stripped.
    """
    f = open(path, encoding="utf-8")
    try:
        for line in f:
            stripped = line.strip()
            if stripped:
                yield stripped
    finally:
        f.close()
