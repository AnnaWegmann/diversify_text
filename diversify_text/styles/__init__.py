"""Style bank data and style dict resolving

``bank`` is the API for the built in style data
 ``resolve`` turns the user-facing style parameters into the canonical style dict.
"""

from diversify_text.styles.bank import DEFAULT_STYLE_BANK
from diversify_text.styles.resolve import resolve_style_dict

__all__ = [
    "DEFAULT_STYLE_BANK",
    "resolve_style_dict",
]
