"""Resolution of user-facing style parameters into the canonical style dict.

The resolver follows the parse-don't-validate pattern: the several input
shapes accepted by :func:`~diversify_text.core.diversify` are normalized
once, here at the boundary, into one ordered ``dict[str, list[str]]``
(style name → example texts) that all downstream code consumes.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from diversify_text.styles.bank import DEFAULT_STYLE_BANK, DEFAULT_STYLES

logger = logging.getLogger(__name__)

#: Suffix appended to a user style whose name clashes with a selected bank style.
_USER_SUFFIX = "_user"


def resolve_style_dict(
    styles: list[str | int] | None = None,
    style_examples: list[str] | list[list[str]] | dict[str, list[str]] | None = None,
    *,
    bank: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Resolve *styles* and *style_examples* into one ordered style dict.

    Parameters
    ----------
    styles : list of str or int, optional
        Selection from the built-in style bank, by name (``"recipe"``)
        and/or 0-based index (``7``).  Mixing is allowed.
    style_examples : list[str] | list[list[str]] | dict, optional
        User-defined styles.  A flat list of strings is one style; a
        list of lists is several styles; a dict maps style names to
        example texts.  Unnamed styles are auto-named ``style_1``,
        ``style_2``, ... (1-based).
    bank : dict, optional
        The style bank to select from.  ``None`` uses
        :data:`~diversify_text.styles.bank.DEFAULT_STYLE_BANK`.

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
    if styles is None and style_examples is None:
        raise ValueError(
            "Provide styles (bank selection) and/or style_examples "
            "(your own example texts)."
        )
    if bank is None:
        bank = DEFAULT_STYLE_BANK

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
            else:
                if entry not in bank:
                    raise ValueError(
                        f"Unknown style {entry!r}. "
                        f"Available: {sorted(bank)}"
                    )
                name = entry
            if name in resolved:
                raise ValueError(
                    f"Style {name!r} requested more than once in styles."
                )
            resolved[name] = list(bank[name])

    # --- user styles (style_examples) ---
    if style_examples is not None:
        for name, examples in _normalize_style_examples(style_examples).items():
            if not examples:
                raise ValueError(f"Style {name!r} has no example texts.")
            if name in resolved:
                renamed = f"{name}{_USER_SUFFIX}"
                if renamed in resolved:
                    raise ValueError(
                        f"Style name {name!r} clashes with a selected bank "
                        f"style and {renamed!r} is also taken. Rename your "
                        "style in style_examples."
                    )
                logger.warning(
                    "Style name %r is also a selected bank style; "
                    "renaming your style to %r.",
                    name, renamed,
                )
                name = renamed
            resolved[name] = examples

    return resolved


def _normalize_style_examples(
    style_examples: list[str] | list[list[str]] | dict[str, list[str]],
) -> dict[str, list[str]]:
    """Normalize the three accepted *style_examples* shapes into a dict.

    Auto-names unnamed styles ``style_1``, ``style_2``, ... (1-based).
    """
    if isinstance(style_examples, Mapping):
        result: dict[str, list[str]] = {}
        for name, examples in style_examples.items():
            if not isinstance(examples, list) or not all(
                isinstance(x, str) for x in examples
            ):
                raise TypeError(
                    f"style_examples[{name!r}] must be a list of strings."
                )
            result[str(name)] = list(examples)
        return result

    if isinstance(style_examples, list):
        # Flat list of strings → one style.
        if all(isinstance(x, str) for x in style_examples):
            return {"style_1": list(style_examples)}
        # List of lists of strings → several styles.
        if all(
            isinstance(group, list)
            and all(isinstance(x, str) for x in group)
            for group in style_examples
        ):
            return {
                f"style_{i}": list(group)
                for i, group in enumerate(style_examples, start=1)
            }
        raise TypeError(
            "style_examples list entries must be all strings (one style) "
            "or all lists of strings (several styles), not a mix."
        )

    raise TypeError(
        "style_examples must be a list of strings, a list of lists of "
        "strings, or a dict mapping style names to lists of strings."
    )


def resolve_style_sets(
    style_bank: dict[str, list[str]] | None = None,
    styles: list[str] | None = None,
) -> dict[str, list[str]]:
    """Resolve style bank and optional key filter into a style dict.

    Used by both TinyStyler and the prompting method.

    Parameters
    ----------
    style_bank : dict or None
        Custom style bank. ``None`` falls back to
        :data:`DEFAULT_STYLE_BANK`.
    styles : list[str] or None
        Select only these keys from the bank. Order is preserved.

    Returns
    -------
    dict[str, list[str]]
        Mapping of style names to example sentence lists.
    """
    bank = style_bank if style_bank is not None else DEFAULT_STYLE_BANK

    if styles is not None:
        unknown = set(styles) - set(bank.keys())
        if unknown:
            raise ValueError(
                f"Unknown style key(s): {sorted(unknown)}. "
                f"Available: {sorted(bank.keys())}"
            )
        return {k: bank[k] for k in styles}

    if style_bank is not None:
        return dict(style_bank)

    return {k: bank[k] for k in DEFAULT_STYLES}
