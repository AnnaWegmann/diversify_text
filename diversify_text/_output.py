"""Output path resolution and incremental writing for diversify.

Decides *where* results go (in-memory vs. disk) and writes them in the
appropriate format (Python list or JSONL).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import IO, Any, Union

from diversify_text._input import InputContext, InputKind

_log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Type alias
# ------------------------------------------------------------------

DiversifyOutput = Union[list[dict], Path]


# ------------------------------------------------------------------
# Output path resolution
# ------------------------------------------------------------------


def resolve_output_path(
    input_context: InputContext,
    output_dir: str | Path | None = None,
    output_name: str | None = None,
) -> Path | None:
    """Determine where output should be written, or ``None`` for in-memory.

    The user controls *where* (directory) and *what name* (stem) to use,
    but the **extension is always ``.jsonl``**.

    Directory defaults (when *output_dir* is ``None``):

    * ``SINGLE_STR`` / ``LIST`` → ``None`` (keep in memory, return as
      ``list[dict]``).  These are small, known-size inputs from Python
      code, so the caller typically wants results as Python objects.
    * ``ITERABLE`` → current working directory.
    * ``FILE_CSV`` / ``FILE_TSV`` / ``FILE_TXT`` → same directory as
      the input file.

    If *output_dir* is provided for ``SINGLE_STR`` / ``LIST``, results
    are written to disk instead of being returned in memory.

    Name defaults (when *output_name* is ``None``):

    * ``FILE_CSV`` / ``FILE_TSV`` → ``<input_stem>_diversified``
    * ``FILE_TXT`` → ``<input_stem>``
    * Everything else → ``diversified_output``

    Parameters
    ----------
    input_context : InputContext
        Metadata produced by :func:`resolve_input`.
    output_dir : str, Path, or None
        Directory to write output files into.
    output_name : str or None
        Base filename (without extension).  The correct extension is
        appended automatically.  If the name already contains an
        extension it is **not** stripped — the correct extension is
        appended after it — unless it already ends with ``.jsonl``.

    Returns
    -------
    Path or None
        ``None`` means in-memory mode; otherwise the path to write to.
    """
    # --- determine directory ---
    if output_dir is not None:
        directory = Path(output_dir)
    elif input_context.kind in (InputKind.SINGLE_STR, InputKind.LIST):
        # No output_dir and in-memory input → stay in-memory.
        return None
    elif input_context.kind == InputKind.ITERABLE:
        # Iterable with no output_dir → default to current working directory.
        directory = Path.cwd()
    else:
        # FILE_CSV, FILE_TSV, FILE_TXT → same directory as input file.
        assert input_context.input_path is not None
        directory = input_context.input_path.parent

    # --- determine base name ---
    if output_name is not None:
        name = output_name
    elif input_context.kind in (InputKind.FILE_CSV, InputKind.FILE_TSV):
        assert input_context.input_path is not None
        name = f"{input_context.input_path.stem}_diversified"
    elif input_context.kind == InputKind.FILE_TXT:
        assert input_context.input_path is not None
        name = input_context.input_path.stem
    else:
        # ITERABLE, or LIST/SINGLE_STR with output_dir.
        name = "diversified_output"

    # --- build final path with the correct extension ---
    if not name.endswith(".jsonl"):
        name = f"{name}.jsonl"
    result = directory / name

    _log.info("Output will be written to %s", result)
    return result


# ------------------------------------------------------------------
# Output writer
# ------------------------------------------------------------------


class OutputWriter:
    """Incrementally writes diversify results to the right format.

    Modes
    -----
    * **In-memory** (``output_path is None``): accumulates
      ``list[dict]`` with keys ``"original"`` and ``"paraphrases"``.
    * **JSONL** (``output_path is not None``): writes one JSON object
      per line to a ``.jsonl`` file.
    """

    def __init__(
        self,
        input_context: InputContext,
        n: int,
        output_path: Path | None,
    ) -> None:
        """Initialize the writer.

        Parameters
        ----------
        input_context : InputContext
            Metadata about the input source (kind, path, etc.).
        n : int
            Number of paraphrase styles requested per text.
        output_path : Path or None
            Where to write results on disk.  ``None`` means results
            are kept in memory and returned as ``list[dict]``.
        """
        self._input_context = input_context
        self._n = n
        self._output_path = output_path
        # Open file handle — set by open() when writing to disk.
        self._handle: IO[str] | None = None
        # In-memory accumulator — used only when output_path is None.
        self._accumulated: list[dict[str, Any]] = []

    # --- lifecycle: open / write / close ---

    def open(self) -> None:
        """Open the file handle when writing to disk.

        Must be called before :meth:`write_batch`.

        * ``output_path is None`` — does nothing (in-memory mode).
        * Otherwise — opens a single JSONL file for writing.
        """
        if self._output_path is None:
            # In-memory mode: nothing to open.
            return

        # Disk mode: open a single JSONL file.
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self._output_path, "w", encoding="utf-8")

    def write_batch(
        self,
        originals: list[str],
        paraphrases_by_text: list[list[dict[str, str]]],
    ) -> None:
        """Append one batch of results.

        Parameters
        ----------
        originals : list[str]
            The original texts in this batch.
        paraphrases_by_text : list[list[dict[str, str]]]
            One inner list per original text, each containing one
            ``{"style": ..., "text": ...}`` entry per target style.
            For example, with 2 styles and 1 text:
            ``[[{"style": "opinion", "text": "..."},
            {"style": "scottish_english", "text": "..."}]]``.

        Raises
        ------
        ValueError
            If ``originals`` and ``paraphrases_by_text`` have different
            lengths.
        """
        if len(originals) != len(paraphrases_by_text):
            raise ValueError(
                f"originals has {len(originals)} items but "
                f"paraphrases_by_text has {len(paraphrases_by_text)}."
            )

        for i, (orig, paras) in enumerate(zip(originals, paraphrases_by_text)):
            if len(paras) != self._n:
                _log.debug(
                    "Expected %d paraphrases for text %d, got %d.",
                    self._n, i, len(paras),
                )
            record = {"original": orig, "paraphrases": paras}
            if self._output_path is None:
                self._accumulated.append(record)
            else:
                assert self._handle is not None
                self._handle.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

    def finish(self) -> DiversifyOutput:
        """Close the file handle and return the final result.

        Returns
        -------
        list[dict]
            When ``output_path`` was ``None`` (in-memory mode).  Each dict
            has keys ``"original"`` and ``"paraphrases"``.
        Path
            When results were written to disk — the ``.jsonl`` path.
        """
        if self._handle is not None:
            self._handle.close()
            self._handle = None

        if self._output_path is None:
            # In-memory mode: return the accumulated list of dicts.
            return self._accumulated

        # Disk mode: return the output path.
        return self._output_path
