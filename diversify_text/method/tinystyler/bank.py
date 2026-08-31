"""TinyStyler's own style banks.

TinyStyler transfers formality and social-media voice well, but not
dialects, historical English, or most genres, so it does not use the
package's default style bank.

``tinystylerbank.json`` (in this directory) holds the example texts
that are not in ``stylebank.json``.  Origin of those styles:
``informal``, ``question``, and ``obama`` come from TinyStyler's own
documentation, ``formal`` from STEL, the rest from the CORE corpus.
The dict order below is the bank order (``n`` selects from the front).
"""

from __future__ import annotations

from diversify_text.styles import DEFAULT_STYLE_BANK, SURFACE_STYLE_BANK
from diversify_text.styles._load import load_style_bank

_TS_json = load_style_bank(
    "tinystylerbank.json", package="diversify_text.method.tinystyler"
)

TINYSTYLER_STYLE_BANK: dict[str, list[str]] = {
    "informal": _TS_json["informal"],
    "formal": _TS_json["formal"],
    "question": _TS_json["question"],
    "question_answer_forum": _TS_json["question_answer_forum"],
    "discussion_forum": _TS_json["discussion_forum"],
    "obama": _TS_json["obama"],
    "formal_speech": _TS_json["formal_speech"],
    "personal_blog": _TS_json["personal_blog"],
    "song_lyrics": _TS_json["song_lyrics"],
    "ddlovato": DEFAULT_STYLE_BANK["ddlovato"],
    "britneyspears": DEFAULT_STYLE_BANK["britneyspears"],
}

#: Styles that work with TinyStyler but are more likely to produce
#: swearing.  Selectable by name only.
TINYSTYLER_UNUSUAL_STYLE_BANK: dict[str, list[str]] = {
    "spoken_communication": DEFAULT_STYLE_BANK["spoken_communication"],
    "digital_communication": DEFAULT_STYLE_BANK["digital_communication"],
    "other_spoken": _TS_json["other_spoken"],
    "reader_viewer_responses": _TS_json["reader_viewer_responses"],
    "arianagrande": DEFAULT_STYLE_BANK["arianagrande"],
}

#: The surface styles TinyStyler handles reasonably.
TINYSTYLER_SURFACE_STYLE_BANK: dict[str, list[str]] = {
    name: SURFACE_STYLE_BANK[name]
    for name in (
        "texting_abbreviations",
        "exclamations",
        "lowercase",
        "no_punctuation",
        "all_caps",
    )
}

_used = set(TINYSTYLER_STYLE_BANK) | set(TINYSTYLER_UNUSUAL_STYLE_BANK)
if not set(_TS_json) <= _used:
    raise ValueError(
        "tinystylerbank.json and the banks in "
        "diversify_text/method/tinystyler/bank.py are out of sync: "
        f"unused styles in the JSON: {sorted(set(_TS_json) - _used)}."
    )
