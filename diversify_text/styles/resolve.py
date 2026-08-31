"""Resolution of user-facing style parameters into one normalized style dict.

:func:`~diversify_text.core.diversify` accepts styles in several shapes
(bank names, bank indices, a user's own example texts as list or dict).
The functions here convert all of those, once, into a single ordered
``dict[str, list[str]]`` mapping each style name to its example texts.
All later code only ever sees that dict, so shape handling happens in
exactly one place and bad input fails early with a clear message.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from diversify_text.styles.bank import DEFAULT_STYLE_BANK, UNUSUAL_STYLE_BANK

logger = logging.getLogger(__name__)

#: Suffix appended to a user style whose name clashes with a selected bank style.
_USER_SUFFIX = "_user"


def resolve_style_dict(
    styles: list[str | int] | None = None,
    style_texts: list[str] | list[list[str]] | dict[str, list[str]] | None = None,
    *,
    bank: dict[str, list[str]] | None = None,
    unusual_bank: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Resolve *styles* and *style_texts* into one ordered style dict.

    Parameters
    ----------
    styles : list of str or int, optional
        Selection from the built-in style bank, by name (``"scottish_english"``)
        and/or 0-based index (``7``).  Mixing is allowed.
    style_texts : list[str] | list[list[str]] | dict, optional
        User-defined styles.  A flat list of strings is one style; a
        list of lists is several styles; a dict maps style names to
        example texts.  Unnamed styles are auto-named ``style_1``,
        ``style_2``, ... (1-based).
    bank : dict, optional
        The style bank to select from.  ``None`` uses
        :data:`~diversify_text.styles.bank.DEFAULT_STYLE_BANK`.
    unusual_bank : dict, optional
        Additional, unusual styles selectable by *name* only.
        They are not part of the default pool.
        When *bank* is ``None`` this defaults to
        :data:`~diversify_text.styles.bank.UNUSUAL_STYLE_BANK`.
        If the user provides their own bank this defaults to None.

    Returns
    -------
    dict[str, list[str]]
        Ordered mapping of style name → example texts: bank selections
        first (in *styles* order), then user styles.

    Raises
    ------
    ValueError
        For unknown names, out-of-range indices, a bank style requested
        twice, a style with no example texts, an unresolvable name
        clash, or when neither parameter is provided.
    TypeError
        For input shapes that match none of the accepted forms.
    """
    if styles is None and style_texts is None:
        raise ValueError(
            "Provide styles (bank selection) and/or style_texts "
            "(your own example texts)."
        )
    if bank is None:
        bank = DEFAULT_STYLE_BANK
        if unusual_bank is None:
            unusual_bank = UNUSUAL_STYLE_BANK
    if unusual_bank is None:
        unusual_bank = {}

    resolved: dict[str, list[str]] = {}

    # --- bank selection (styles) ---
    if styles is not None:
        bank_names = list(bank.keys())
        for entry in styles:
            # bool is an int subclass — reject it explicitly.
            if isinstance(entry, bool) or not isinstance(entry, (int, str)):
                raise TypeError(
                    "styles entries must be str names or int indices, "
                    f"got {entry!r}."
                )
            if isinstance(entry, int):
                if not 0 <= entry < len(bank_names):
                    raise ValueError(
                        f"Style index {entry} is out of range. The bank "
                        f"has {len(bank_names)} styles "
                        f"(indices 0-{len(bank_names) - 1})."
                    )
                name = bank_names[entry]
                source = bank
            elif entry in bank:
                name = entry
                source = bank
            elif entry in unusual_bank:
                name = entry
                source = unusual_bank
            else:
                by_name_only = (
                    f" By name only: {sorted(unusual_bank)}."
                    if unusual_bank else ""
                )
                raise ValueError(
                    f"Unknown style {entry!r}. "
                    f"Available: {sorted(bank)}.{by_name_only}"
                )
            if name in resolved:
                raise ValueError(
                    f"Style {name!r} requested more than once in styles."
                )
            resolved[name] = list(source[name])

    # --- user styles (style_texts) ---
    if style_texts is not None:
        for name, examples in _normalize_style_texts(style_texts).items():
            if not examples:
                raise ValueError(f"Style {name!r} has no example texts.")
            if name in resolved:
                renamed = f"{name}{_USER_SUFFIX}"
                if renamed in resolved:
                    raise ValueError(
                        f"Style name {name!r} clashes with a selected bank "
                        f"style and {renamed!r} is also taken. Rename your "
                        "style in style_texts."
                    )
                logger.warning(
                    "Style name %r is also a selected bank style; "
                    "renaming your style to %r.",
                    name, renamed,
                )
                name = renamed
            resolved[name] = examples

    return resolved


def _normalize_style_texts(
    style_texts: list[str] | list[list[str]] | dict[str, list[str]],
) -> dict[str, list[str]]:
    """Normalize the three accepted *style_texts* shapes into a dict.

    Auto-names unnamed styles ``style_1``, ``style_2``, ... (1-based).
    """
    if isinstance(style_texts, Mapping):
        result: dict[str, list[str]] = {}
        for name, examples in style_texts.items():
            if not isinstance(examples, list) or not all(
                isinstance(x, str) for x in examples
            ):
                raise TypeError(
                    f"style_texts[{name!r}] must be a list of strings."
                )
            result[str(name)] = list(examples)
        return result

    if isinstance(style_texts, list):
        # Flat list of strings → one style.
        if all(isinstance(x, str) for x in style_texts):
            return {"style_1": list(style_texts)}
        # List of lists of strings → several styles.
        if all(
            isinstance(group, list)
            and all(isinstance(x, str) for x in group)
            for group in style_texts
        ):
            return {
                f"style_{i}": list(group)
                for i, group in enumerate(style_texts, start=1)
            }
        raise TypeError(
            "style_texts list entries must be all strings (one style) "
            "or all lists of strings (several styles), not a mix."
        )

    raise TypeError(
        "style_texts must be a list of strings, a list of lists of "
        "strings, or a dict mapping style names to lists of strings."
    )


