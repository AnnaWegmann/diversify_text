"""Input resolution and output writing for diversify.

All I/O concerns live here so that ``core.py`` only works with plain
``list[str]`` batches.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import IO, Any, Union


# ------------------------------------------------------------------
# Type aliases (single source of truth)
# ------------------------------------------------------------------

TextInput = Union[str, "list[str]", "Iterable[str]"]
DiversifyOutput = Union[list[dict], Path]


# ------------------------------------------------------------------
# Input resolution
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
            rows = _read_csv(path, text_column, delimiter=",")
            return iter(rows), InputContext(
                kind=InputKind.FILE_CSV,
                input_path=path,
                text_column=text_column,
                total=len(rows),
            )

        if suffix == ".tsv" and path.is_file():
            rows = _read_csv(path, text_column, delimiter="\t")
            return iter(rows), InputContext(
                kind=InputKind.FILE_TSV,
                input_path=path,
                text_column=text_column,
                total=len(rows),
            )

        if suffix == ".txt" and path.is_file():
            lines = _read_txt_lines(path)
            return iter(lines), InputContext(
                kind=InputKind.FILE_TXT,
                input_path=path,
                total=len(lines),
            )

        # Not a recognised file — treat as a single text.
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
# File readers
# ------------------------------------------------------------------


def _read_csv(path: Path, text_column: str, delimiter: str) -> list[str]:
    """Read a CSV/TSV file and return the *text_column* values."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None or text_column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames or [])
            raise ValueError(
                f"Column '{text_column}' not found in {path}. "
                f"Available: {available}"
            )
        return [row.get(text_column) or "" for row in reader]


def _read_txt_lines(path: Path) -> list[str]:
    """Read a ``.txt`` file, returning one text per non-empty line."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ------------------------------------------------------------------
# Output path resolution
# ------------------------------------------------------------------


def resolve_output_path(
    ctx: InputContext,
    output_path: str | Path | None,
) -> Path | None:
    """Determine where output should be written, or *None* for in-memory.

    Raises
    ------
    ValueError
        When *ctx.kind* is ``ITERABLE`` and no *output_path* is provided.
    """
    if output_path is not None:
        return Path(output_path)

    if ctx.kind in (InputKind.SINGLE_STR, InputKind.LIST):
        return None

    if ctx.kind == InputKind.ITERABLE:
        raise ValueError(
            "output_path is required when input is an iterator/generator."
        )

    if ctx.kind == InputKind.FILE_TXT:
        # Base path — OutputWriter derives _diversified_N.txt from this.
        assert ctx.input_path is not None
        return ctx.input_path

    if ctx.kind in (InputKind.FILE_CSV, InputKind.FILE_TSV):
        assert ctx.input_path is not None
        return ctx.input_path.with_name(
            f"{ctx.input_path.stem}_diversified.jsonl"
        )

    return None  # unreachable, but keeps mypy happy


# ------------------------------------------------------------------
# Output writer
# ------------------------------------------------------------------


class OutputWriter:
    """Incrementally writes diversification results to the right format.

    Modes
    -----
    * **In-memory** (``output_path is None``): accumulates
      ``list[dict]`` with keys ``"original"`` and ``"paraphrases"``.
    * **JSONL** (CSV/TSV/iterable file input): writes one JSON object
      per line.
    * **TXT multi-file** (``.txt`` input): opens *n_styles* files
      (``<stem>_diversified_1.txt``, …) and writes the i-th paraphrase
      of each text to the i-th file.
    """

    def __init__(
        self,
        ctx: InputContext,
        n_styles: int,
        output_path: Path | None,
    ) -> None:
        self._ctx = ctx
        self._n_styles = n_styles
        self._output_path = output_path
        self._handles: list[IO[str]] = []
        self._accumulated: list[dict[str, Any]] = []

    # --- lifecycle ---

    def open(self) -> None:
        """Open file handles when writing to disk."""
        if self._output_path is None:
            return

        if self._ctx.kind == InputKind.FILE_TXT:
            base = self._output_path
            for i in range(1, self._n_styles + 1):
                p = base.with_name(f"{base.stem}_diversified_{i}.txt")
                p.parent.mkdir(parents=True, exist_ok=True)
                self._handles.append(open(p, "w", encoding="utf-8"))
        else:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            self._handles.append(
                open(self._output_path, "w", encoding="utf-8")
            )

    def write_batch(
        self,
        originals: list[str],
        paraphrases_by_text: list[list[str]],
    ) -> None:
        """Append one batch of results."""
        if self._output_path is None:
            for orig, paras in zip(originals, paraphrases_by_text):
                self._accumulated.append(
                    {"original": orig, "paraphrases": paras}
                )
            return

        if self._ctx.kind == InputKind.FILE_TXT:
            for orig, paras in zip(originals, paraphrases_by_text):
                for i, handle in enumerate(self._handles):
                    text = paras[i] if i < len(paras) else ""
                    handle.write(text + "\n")
        else:
            handle = self._handles[0]
            for orig, paras in zip(originals, paraphrases_by_text):
                record = {"original": orig, "paraphrases": paras}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def finish(self) -> DiversifyOutput:
        """Close file handles and return the final result."""
        for h in self._handles:
            h.close()
        self._handles.clear()

        if self._output_path is None:
            return self._accumulated

        if self._ctx.kind == InputKind.FILE_TXT:
            return self._output_path
        return self._output_path
