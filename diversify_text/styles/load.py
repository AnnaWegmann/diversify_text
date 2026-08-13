"""Loading of style data from ``stylebank.json``.

``stylebank.json`` includes a curated dict of the used style taxonomy.
This loader flattens the nests styles in the linguistic taxonomy
(``language_variation → intra-group → diatopic → welsh_english → [examples]``)
to ``name → examples`` dict. All shaping happens here at load time.
"""

from __future__ import annotations

import json
from importlib import resources

#: Leaf names become public style names verbatim, except the few with
#: characters beyond ``[a-z0-9_-]``, which are cleaned up here.  Renaming
#: at load time keeps ``stylebank.json`` untouched; the JSON key is only
#: ever seen through this mapping, so a rename here changes the public
#: style name.
_RENAMES: dict[str, str] = {
    "barbadian_creole_(bajan)": "barbadian_creole",
    "education_somehighschool,nodiploma": "education_some_highschool_no_diploma",
}

_STYLEBANK_FILENAME = "stylebank.json"


def load_style_bank() -> dict[str, list[str]]:
    """Load and flatten the style bank shipped with the package.

    Returns
    -------
    dict[str, list[str]]
        Flat ordered mapping of style name → example texts, in the
        file's traversal order (the curated bank order is applied by the
        caller, not here).
    """
    # importlib.resources (not __file__) so the JSON is also found when
    # the package is installed as a wheel/zip.
    raw = (
        resources.files("diversify_text.styles")
        .joinpath(_STYLEBANK_FILENAME)
        .read_text(encoding="utf-8")
    )
    return flatten_style_bank(json.loads(raw))


def flatten_style_bank(nested: dict) -> dict[str, list[str]]:
    """Flatten a nested style taxonomy into a ``name → examples`` dict.

    Inner dicts are taxonomy levels; a list value is a leaf holding one
    style's example texts.  Only the leaf name survives flattening, so
    leaf names must be unique across the whole taxonomy.

    Raises
    ------
    ValueError
        For duplicate leaf names, a leaf that is not a non-empty list of
        strings, or a taxonomy node that is neither dict nor list.
    """
    flat: dict[str, list[str]] = {}
    _walk(nested, path=(), flat=flat)
    return flat


def _walk(node: dict, path: tuple[str, ...], flat: dict[str, list[str]]) -> None:
    for key, value in node.items():
        here = path + (key,)
        if isinstance(value, dict):
            _walk(value, here, flat)
            continue
        if not isinstance(value, list):
            raise ValueError(
                f"Invalid style bank entry at {' → '.join(here)}: expected "
                "a taxonomy dict or a list of example texts, got "
                f"{type(value).__name__}."
            )
        if not value or not all(isinstance(x, str) for x in value):
            raise ValueError(
                f"Style {' → '.join(here)!s} must be a non-empty list "
                "of strings."
            )
        name = _RENAMES.get(key, key)
        if name in flat:
            raise ValueError(
                f"Duplicate style name {name!r} (at {' → '.join(here)}); "
                "leaf names must be unique across the taxonomy."
            )
        flat[name] = list(value)
